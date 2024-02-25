# To run this BACKEND server locally, follow the following steps. 

- Clone by running  `git clone https://github.com/AbdullahHabib-github/ADHD` in the terminal/console.
- Run the following commands in the terminal/console
    1. `cd Backend`
    2. `python -m venv venv`
    3. `venv/Scripts/activate`
    4. `pip install -r requirements.txt`
- Adding Api Key
    1. Get your api key for Gemini from [Ai Studio](https://aistudio.google.com/app/apikey)
    2. Paste your Gemini api key in the the [google_api.txt](Backend/google_api.txt) file.

- Add Credentials for Google Cloud Console
    1. Follow the steps at [GCP Documentations](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev) to set up credentials for Google Cloud Platform.
    2. Get your Credentials (json file) from Goolge Cloud Console.
    3. Place the json file in the backend base directory [Base Directory](Backend)
- Run `python app.py` in  the terminal/console to start the application.
- The local server is now live.
