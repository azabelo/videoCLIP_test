from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.io import read_video
from pytube import YouTube
from mmpt.models import MMPTModel

# Define the YouTube video URL
video_url = "https://www.youtube.com/watch?v=6XOCSyJ2kwA"

# Define the file path for saving the downloaded video
output_file = "test_yt.mp4"

# Download the video using pytube
yt = YouTube(video_url)
yt.streams.filter(progressive=True, file_extension="mp4").first().download(output_path=".", filename="downloaded_video.mp4")

# Read the downloaded video file into a PyTorch tensor
video, audio, info = read_video("downloaded_video.mp4", pts_unit="sec")
video_tensor = video.permute(3, 0, 1, 2)  # Rearrange dimensions to (C, T, H, W)

# Resize video frames to 224x224
video_frames = []
for frame in video:
    # Convert the frame to RGB format

    frame_rgb = frame.permute(1, 2, 0).numpy().astype('uint8')
    img = Image.fromarray(frame_rgb, 'RGB')
    img = F.resize(img, (224, 224))
    img = F.to_tensor(img)
    video_frames.append(img)
resized_video_tensor = torch.stack(video_frames, dim=0)
print(resized_video_tensor.shape)

# Loop through the frames and display them
for frame in resized_video_tensor:
    # Convert the frame tensor to a NumPy array
    frame_np = frame.permute(1, 2, 0).numpy()

    # Scale the frame values to [0, 255] and ensure the data type is 'uint8'
    frame_scaled = (frame_np * 255).clip(0, 255).astype("uint8")

    # Display the frame in the "Video" window

    # cv2.imshow("Video", frame_scaled)

resized_video_tensor = resized_video_tensor[:240, :, :, :].unsqueeze(0)  # Add batch dimension

# Print some information about the video
print(f"Video Shape: {resized_video_tensor.shape}")



model, tokenizer, aligner = MMPTModel.from_pretrained(
    "projects/retri/videoclip/how2.yaml")

model.eval()


# B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
video_frames = resized_video_tensor.view(1, 8, 30, 3, 224, 224)
video_frames = video_frames.permute(0, 1, 2, 4, 5, 3)
print(f"new video Shape: {video_frames.shape}")
caps, cmasks = aligner._build_text_seq(
    tokenizer("dog playing fetch", add_special_tokens=False)["input_ids"]
)

caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

with torch.no_grad():
    output = model(video_frames, caps, cmasks, return_score=True)
print(output["score"])  # dot-product
