**Prerequisites**

Python 3.8 or higher
Required libraries (can be installed via pip install -r requirements.txt)



**Usage**
Run the script using the following command:
```
python main.py
```


***How It Works***

Create a Task:
The create_task function sends a POST request to create a task.
On success, it retrieves and logs the taskId.

Poll Task Status:
The process_task function calls get_task to query the status of the created task.

The status can be:

Working: The task is in progress. It will retry after a specified interval.

Success: The task is completed successfully. The result is logged.

Failed: The task failed. The script logs the failure and exits.
Unknown status: Logs the unknown status and exits.

Retry Logic:
If the task is Working, it retries for a maximum of MAX_RETRIES (default: 10) with a RETRY_INTERVAL of 3 seconds between attempts.