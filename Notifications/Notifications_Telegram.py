#________________________________Notification Experiment________________________#

import requests
import os
from datetime import datetime

class NotificationExperiment():
    def __init__(self, bot_token, chat_id, image_path):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_notification(self, image_path, title, message):
        caption = f"*{title}*\n\n{message}\n\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        url = f"{self.base_url}/sendPhoto"
        try:
            with open(image_path, 'rb') as image_file:
                files = {
                    'photo': image_file
                }
                
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                print(f"Image with text sent! Message ID: {result['result']['message_id']}")
                return result
            else:
                print(f"Failed to send image: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error sending image: {e}")
if __name__ == "__main__":
  print("Starting Notification Experiment ==== Image with Text Payload ====")
  bot_token= ###########
  chat_id= ############
  image_path = #########
    mobile_notification = NotificationExperiment(bot_token, chat_id, image_path)
    mobile_notification.send_notification(
        image_path,
        title="Notification Experiment",
        message="This is a test notification with an image and text payload."
    )
