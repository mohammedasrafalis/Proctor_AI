from twilio.rest import Client
from django.conf import settings

def send_whatsapp_message(to_number, message):
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    return client.messages.create(
        body=message,
        from_='whatsapp:+14155238886',  # Twilio sandbox number
        to=f'whatsapp:{to_number}'
    )
