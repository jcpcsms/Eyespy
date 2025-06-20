sudo docker run -it --rm \
  --name jetson-server \
  --runtime nvidia \
  --network host \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video1:/dev/video1 \
  -v /home/jcp/multi-camera-action-recognition:/workspace \
  jetson-server-61825:latest
