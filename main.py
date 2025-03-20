import os
import cv2
import typer
import numpy as np
import supervision as sv
from ultralytics import YOLO
import google.generativeai as genai
import textwrap
from PIL import ImageFont, ImageDraw, Image  

# Set up API Key for Gemini
os.environ["API_KEY"] = ""
genai.configure(api_key=os.environ["API_KEY"])


gemini_model = genai.GenerativeModel("gemini-2.0-flash")


yolo_model = YOLO("model/cnn_2cls_ver2.engine")


app = typer.Typer()

# Process webcam or video file with YOLO and Gemini
def process_webcam(output_file="output.mp4", frame_rate=240):
    cap = cv2.VideoCapture("data/chero.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    target_width = 1080
    target_height = 720


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_file,
        fourcc,
        fps,
        (target_width, target_height)
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    frame_count = 0
    frame_to_analyze = None
    analysis_text = "Đang phân tích video..."
    
    # Load font
    font_path = "Roboto-LightItalic.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_width, target_height))

        results = yolo_model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Tạo annotated frame
        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )

        # Xử lý phân tích với Gemini sau mỗi frame_rate frames
        if frame_count % frame_rate == 0:
            frame_to_analyze = annotated_frame.copy()
        elif frame_to_analyze is not None:
            try:
                temp_image_path = "temp_frame.jpg"
                cv2.imwrite(temp_image_path, frame_to_analyze)
                
                image_file = genai.upload_file(temp_image_path)
                prompt = [
                    image_file,
                    """
                    Bạn là một nhà văn tài năng. Hãy mô tả hình ảnh này như thể bạn đang kể lại một câu chuyện sống động, đầy cảm xúc và sáng tạo.
                    Hãy chú ý đến các chi tiết nhỏ, màu sắc, không gian và cảm giác tổng thể của cảnh vật.
                    Văn phong nhẹ nhàng, tả thực, lôi cuốn.
                    Độ dài mô tả từ 40 đến 80 từ.
                    """
                ]
                
                response = gemini_model.generate_content(prompt)
                analysis_text = response.text.strip()
                
                os.remove(temp_image_path)
                frame_to_analyze = None
                
            except Exception as e:
                print(f"Lỗi khi phân tích Gemini: {e}")
                analysis_text = "Phân tích thất bại."
                frame_to_analyze = None

        # Thêm background mờ cho text
        text_background_height = 150
        text_background = annotated_frame[-text_background_height:, :].copy()
        text_background = cv2.GaussianBlur(text_background, (25, 25), 0)
        annotated_frame[-text_background_height:, :] = text_background

        # Xử lý text tiếng Việt
        wrapped_text = textwrap.wrap(analysis_text, width=100)
        annotated_frame_pil = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        annotated_frame_pil = Image.fromarray(annotated_frame_pil)
        draw = ImageDraw.Draw(annotated_frame_pil)

        for i, line in enumerate(wrapped_text):
            text_position = (10, target_height - text_background_height + 20 + i * (font_size + 5))
            draw.text(text_position, line, font=font, fill=(0, 255, 0))

        annotated_frame = cv2.cvtColor(np.array(annotated_frame_pil), cv2.COLOR_RGB2BGR)

        out.write(annotated_frame)
        cv2.imshow("Video Analysis", annotated_frame)
        
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.command()
def webcam(output_file: str = "output.mp4", frame_rate: int = 240):
    typer.echo("Bắt đầu xử lý webcam/video...")
    process_webcam(output_file, frame_rate)

if __name__ == "__main__":
    app()
