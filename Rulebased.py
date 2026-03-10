import re
import streamlit as st 
def chatbot_response(user_input):
    user_input = user_input.lower()

    QandA = [
        (r"hi|hello|hey", "Hello! Welcome to our store. How can I help you today?"),
        
        (r"order status|track order|where is my order",
         "You can track your order in the 'My Orders' section of your account."),

        (r"shipping|delivery time",
         "Standard shipping takes 3-5 business days."),

        (r"return|refund",
         "You can return items within 30 days of delivery for a full refund."),

        (r"payment|pay|payment methods",
         "We accept credit cards, debit cards, and PayPal."),

        (r"store hours|open|closing",
         "Our customer support is available from 9 AM to 6 PM Monday to Friday."),

        (r"bye|exit|quit",
         "Thank you for visiting our store! Have a great day.")
    ]

    for pattern, response in QandA:
        if re.search(pattern, user_input):
            return response

    return "I'm sorry, I didn't understand that. Could you please rephrase?"


st.title("Customer Support Chatbot(Rule based)")

user_input = st.text_input("You:")

if st.button("Send"):
    response = chatbot_response(user_input)
    st.write("Bot:", response)




#we used regular expression for pattern matching.we can also use string match instaed

#  if "hi" in user_input or "hello" in user_input or "hey" in user_input:
#         return "Hello! Welcome to our store. How can I help you?"

#     elif "order" in user_input or "track" in user_input:
#         return "You can track your order in the 'My Orders' section."

#     elif "shipping" in user_input or "delivery" in user_input:
#         return "Shipping usually takes 3-5 business days."

#     elif "return" in user_input or "refund" in user_input:
#         return "You can return items within 30 days for a full refund."

#     elif "payment" in user_input:
#         return "We accept credit cards, debit cards, and PayPal."

#     elif "bye" in user_input or "exit" in user_input:
#         return "Thanks for visiting! Have a great day."

#     else:
#         return "Sorry, I didn't understand that."