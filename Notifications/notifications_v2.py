################################################///             Pull Latest Image for Notification                   \\\########################################################
import requests
import glob
import os

def screenshot_payload(bot_token, chat_id, title, message):
  # Use latest screenshot
  files = glob.glob(screenshots/Eyespy_capture_*.png")
  latest_file = max(files, key=os.path.getmtime)
  url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
  caption = f"{title}\n\n{message}"
 
  with open(latest_file, 'rb') as photo:
    response = requests.post(url, files={'photo': photo}, data={'chat_id': chat_id, 'caption': caption})
  # verify response + error handling
  if response.status_code == 200:
    print("Sent to Telegram Server"
  else:
    print(f"Failed: {response.text}")
    
if __name__ == "__main__":
  screenshot_payload(
    bot_token=####, 
    chat_id=#####, 
    title= "EyeSpy+ Alert", 
    message= "HIGH PRIORITY ALERT"
    )
  


