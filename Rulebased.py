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


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Customer Support Chatbot (Rule based)")

# Display chat history one by one
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input area
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get bot response
    response = chatbot_response(user_input)
    
    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)


