import streamlit as st
import time

macbook_decision = """The sentiment analysis indicates that 84% of reviews for the MacBook Air M1 are positive, 
reflecting high user satisfaction. Meanwhile, 12% of reviews are neutral, and only 4% are negative, showing that 
dissatisfaction is quite uncommon. Overall, the MacBook Air M1 is received very favorably by most users. The N-Gram 
Analysis indicates that users see the lack of traditional USB ports as a downside. In contrast, positive reviews 
focus on the MacBook's excellent battery life and user-friendly design."""

iphone_decision = """For the iPhone 11, 75% of reviews are positive, highlighting strong user satisfaction. In 
contrast, 11% of reviews are neutral, and 14% are negative, pointing to some areas of concern among users. Despite 
these negative reviews, the iPhone 11 remains predominantly well-regarded. The N-Gram Analysis reveals that users 
view limited storage options, the screen's refresh rate, and the absence of EarPods in the case as drawbacks. 
Conversely, positive reviews emphasize the iPhone 11's battery life, camera quality, and overall value for money."""

airpods_decision = """The sentiment analysis for the AirPods 2nd Gen shows that 57% of reviews are positive, 
while 43% are negative. This suggests a generally favorable reception, but also significant concerns from a 
substantial portion of users. Overall, while the AirPods 2nd Gen are appreciated, there are notable issues. The 
N-Gram Analysis shows that users have reported issues with AirPods and the case ceasing to function within a month 
and the AirPods frequently falling out of their ears. Despite these issues, positive reviews highlight the impressive 
battery life and sound quality of the AirPods 2nd Gen, and they are strongly recommended by users."""


def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)
