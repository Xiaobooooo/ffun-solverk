import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

API_BASE_URL = "http://127.0.0.1:6699/v2"
API_TOKEN = "123456"

# Headers with Authorization
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

MAX_RETRIES = 10  # Maximum number of retries
RETRY_INTERVAL = 3  # Retry interval in seconds


def create_task():
    """Create a task"""
    url = f"{API_BASE_URL}/tasks"
    payload = {
        "captchaType": "FunCaptcha",
        "siteReferer": "site_referer",
        "siteKey": "site_key_here",
        "data": "blob_value",
        "proxy": "proxy_value",
    }

    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code == 200:
        result_data = response.json()
        logging.info("Task created successfully:")
        logging.info(result_data)
        return result_data.get("taskId")
    else:
        logging.error(f"Failed to create task: {response.status_code}")
        logging.error(response.json())
        return None


def get_task(task_id):
    """Query task status"""
    url = f"{API_BASE_URL}/tasks/{task_id}"

    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch task status: {response.status_code}")
        logging.error(response.json())
        return None


def process_task():
    """Create a task and poll for its status"""
    task_id = create_task()
    if not task_id:
        logging.error("Unable to create task, exiting process.")
        return None

    attempts = 0
    while attempts < MAX_RETRIES:
        result_data = get_task(task_id)
        if not result_data:
            logging.error("Failed to fetch task status, exiting process.")
            return None

        status = result_data.get("status")
        if status == "Working":
            attempts += 1
            logging.info(f"Task is in progress, retrying... (Attempt {attempts}/{MAX_RETRIES})")
            time.sleep(RETRY_INTERVAL)
        elif status == "Success":
            logging.info(f"Task completed successfully! Result: {result_data['response']}")
            return result_data["response"]
        elif status == "Failed":
            logging.error("Task failed: captcha recognition unsuccessful!")
            return None
        else:
            logging.error(f"Unknown status: {status}, exiting process.")
            return None

    logging.error("Max retries reached, task not completed.")
    return None


if __name__ == "__main__":
    process_task()
