
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Product Feedback Analysis",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# 2. DATA INPUT
# =========================================================

reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality."
]

df = pd.DataFrame({
    "id": [f"R{i:02}" for i in range(1, 19)],
    "text": reviews
})

# =========================================================
# 3. SENTIMENT ANALYSIS
# =========================================================

analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text))
)

# =========================================================
# 4. TOPIC CLASSIFICATION
# =========================================================

topics = {
    "R01": "Insulation",
    "R02": "Portability and Design",
    "R03": "Portability and Design",
    "R04": "Leakage and Durability",
    "R05": "Lid and Usability",
    "R06": "Cleaning",
    "R07": "Insulation",
    "R08": "Overall Experience",
    "R09": "Price and Value",
    "R10": "Portability and Design",
    "R11": "Capacity",
    "R12": "Overall Experience",
    "R13": "Lid and Usability",
    "R14": "Overall Experience",
    "R15": "Insulation",
    "R16": "Price and Value",
    "R17": "Lid and Usability",
    "R18": "Overall Experience"
}

df["topic"] = df["id"].map(topics)

# =========================================================
# 5. NAVIGATION BAR
# =========================================================

st.sidebar.title("📊 Product Feedback")
st.sidebar.caption("Lab 8: Sentiment Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Task 1: Run and Describe",
        "Task 2: Improve Qualitative View",
        "Task 3: Audit Sentiment Labels",
        "Task 4: Compare and Communicate"
    ]
)

st.sidebar.divider()
st.sidebar.write("**Course:** Data Visualization")
st.sidebar.write("**Topic:** Qualitative Data and Sentiment Analysis")

# =========================================================
# STUDENT SUBMISSION DETAILS
# =========================================================

st.sidebar.divider()
st.sidebar.subheader("Student Information")

st.sidebar.write("**Name:**")
st.sidebar.write("Zulaikha binti Mohd Azhar")

st.sidebar.write("**Student ID:**")
st.sidebar.write("24000918")

# =========================================================
# PAGE 1: DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("Product Feedback: A Text Summary")

    st.caption(
        "Prepared by: Zulaikha binti Mohd Azhar | Student ID: 24000918"
    )

    st.write(
        "This dashboard analyses customer comments about a water bottle "
        "using VADER sentiment analysis."
    )

    st.divider()

    # Sentiment overview
    st.header("Sentiment Overview")

    sentiment_order = ["Positive", "Neutral", "Negative"]

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Positive Comments", sentiment_counts["Positive"])

    with col2:
        st.metric("Neutral Comments", sentiment_counts["Neutral"])

    with col3:
        st.metric("Negative Comments", sentiment_counts["Negative"])

    # Sentiment chart
    st.subheader("Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    sentiment_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Comments")
    ax.set_title("Customer Sentiment Distribution")
    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Word cloud
    st.header("Word Cloud")

    all_text = " ".join(df["text"]).lower()

    stop_words = set(STOPWORDS) | {
        "bottle", "drinks", "drink", "water"
    }

    wordcloud = WordCloud(
        width=900,
        height=500,
        background_color="white",
        stopwords=stop_words,
        colormap="viridis",
        collocations=False
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Larger words appear more frequently after common words are removed. "
        "Word size represents frequency, not sentiment. "
        "Colour and position do not indicate positive or negative opinions."
    )

    st.divider()

    # Customer comments
    st.header("Customer Comments")

    st.dataframe(
        df[["id", "text", "sentiment", "compound"]],
        width="stretch",
        hide_index=True
    )


# =========================================================
# PAGE 2: TASK 1 - RUN AND DESCRIBE
# =========================================================

elif page == "Task 1: Run and Describe":

    st.title("Task 1: Run and Describe")

    st.write(
        "This section describes the sentiment analysis results "
        "and identifies recurring themes in the customer comments."
    )

    st.divider()

    # 1.1 Sentiment distribution
    st.header("1.1 Sentiment Distribution")

    st.write(
        "The analysis contains 18 customer comments. "
        "The sentiment results are shown below:"
    )

    st.table(
        pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Number of Comments": [9, 6, 3]
        })
    )

    st.write(
        "Positive sentiment is the largest group, with 9 comments. "
        "Negative sentiment follows with 6 comments, while Neutral "
        "sentiment has 3 comments."
    )

    # 1.2 Recurring themes
    st.header("1.2 Recurring Themes")

    st.write(
        "The recurring themes include product insulation, usability, "
        "design, cleaning convenience, and product quality. Customers "
        "discussed whether the bottle keeps drinks cold, whether it is "
        "easy to carry and clean, and whether its parts are durable "
        "and comfortable to use."
    )

    # 1.3 Mixed-tone comment
    st.header("1.3 Mixed-Tone Comment")

    st.info(
        'Comment R07: "Keeps drinks cold, though the paint scratches easily."'
    )

    st.write(
        "VADER classifies this comment as Positive. However, the customer "
        "expresses both a positive opinion about insulation and a negative "
        "opinion about paint durability. This shows that a single sentiment "
        "label may not fully represent a comment containing mixed opinions."
    )


# =========================================================
# PAGE 3: TASK 2 - IMPROVE QUALITATIVE VIEW
# =========================================================

elif page == "Task 2: Improve Qualitative View":

    st.title("Task 2: Improve Qualitative View")

    st.write(
        "This section explains the improvements made to the word cloud "
        "and the effect of removing selected stopwords."
    )

    st.divider()

    # 2.1 Stopword improvement
    st.header("2.1 Stopword Improvement")

    st.write(
        'The word "water" was added to the stopword list because it '
        "appears in the context of the product but provides limited "
        "information about specific customer experiences. Removing it "
        "helps highlight more meaningful words in the word cloud."
    )

    # 2.2 Retaining useful words
    st.header("2.2 Retaining Useful Words")

    st.write(
        'The word "cleaning" was retained because it highlights an '
        "important product usability issue. Removing it would hide "
        "useful feedback about the difficulty of cleaning the bottle."
    )

    # 2.3 Word cloud interpretation
    st.header("2.3 Word Cloud Interpretation")

    st.write(
        "The word cloud provides a visual summary of frequently used words. "
        "Larger words indicate higher frequency after stopwords are removed. "
        "However, word size, colour, and position do not represent sentiment "
        "or whether customers are satisfied with the product."
    )

    # Display improved word cloud
    all_text = " ".join(df["text"]).lower()

    stop_words = set(STOPWORDS) | {
        "bottle", "drinks", "drink", "water"
    }

    wordcloud = WordCloud(
        width=900,
        height=500,
        background_color="white",
        stopwords=stop_words,
        colormap="viridis",
        collocations=False
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Word size represents frequency after stopwords are removed, "
        "not sentiment."
    )


# =========================================================
# PAGE 4: TASK 3 - AUDIT SENTIMENT LABELS
# =========================================================

elif page == "Task 3: Audit Sentiment Labels":

    st.title("Task 3: Audit Sentiment Labels")

    st.write(
        "This section manually reviews six customer comments and "
        "tests how VADER handles negation."
    )

    st.divider()

    # 3.1 Manual review
    st.header("3.1 Manual Review of Six Comments")

    audit_data = {
        "ID": ["R02", "R03", "R01", "R14", "R04", "R13"],

        "Original Comment": [
            "Great size and easy to carry to class.",
            "I love the simple design and sturdy lid.",
            "The bottle keeps my water cold all day.",
            "It works as described and arrived on time.",
            "It leaks in my bag. Very disappointing.",
            "The lid broke after one week. Poor quality."
        ],

        "VADER Sentiment": [
            "Positive", "Positive", "Neutral",
            "Neutral", "Negative", "Negative"
        ],

        "Manual Review": [
            "Reasonable",
            "Reasonable",
            "Questionable",
            "Reasonable",
            "Reasonable",
            "Reasonable"
        ],

        "Explanation": [
            "The customer is happy with the bottle's size and portability.",
            "The customer expresses satisfaction with the design and lid.",
            "Keeping water cold all day is a product benefit, so the neutral label is questionable.",
            "The customer says the product works as expected and arrived on time.",
            "The leaking problem and disappointment indicate negative feedback.",
            "The broken lid and poor quality indicate negative feedback."
        ]
    }

    audit_df = pd.DataFrame(audit_data)

    st.dataframe(
        audit_df,
        width="stretch",
        hide_index=True
    )

    st.write(
        "Most of the selected sentiment labels are reasonable based on "
        "the comments. However, R01 is questionable because the customer "
        "describes a positive product benefit, even though VADER assigns "
        "a Neutral label."
    )

    # 3.2 Negation test
    st.header("3.2 Negation Test")

    test_review = "The bottle is not easy to clean."

    test_sentiment, test_score = label_sentiment(test_review)

    st.write("Modified customer comment:")

    st.info(test_review)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Sentiment",
            test_sentiment
        )

    with col2:
        st.metric(
            "Compound Score",
            f"{test_score:.4f}"
        )

    st.write(
        "The result is Negative because the phrase 'not easy to clean' "
        "expresses dissatisfaction with the cleaning experience. "
        "The compound score is -0.5667."
    )

    # 3.3 VADER limitations
    st.header("3.3 Limitations of VADER")

    st.write(
        "VADER can identify general positive and negative language, "
        "but it may not fully understand context, sarcasm, or mixed opinions. "
        "For example, a comment may praise one product feature while "
        "complaining about another. Therefore, manual checking is useful "
        "to identify questionable sentiment labels."
    )


# =========================================================
# PAGE 5: TASK 4 - COMPARE AND COMMUNICATE
# =========================================================

elif page == "Task 4: Compare and Communicate":

    st.title("Task 4: Compare and Communicate")

    st.write(
        "This section classifies customer comments by topic, compares "
        "sentiment across topics, and presents a summary for the product team."
    )

    st.divider()

    # 4.1 Topic classification
    st.header("4.1 Main Topic for Each Comment")

    st.write(
        "Each customer comment was assigned a main topic based on "
        "the product feature or experience being discussed."
    )

    st.dataframe(
        df[["id", "text", "topic", "sentiment"]],
        width="stretch",
        hide_index=True
    )

    # 4.2 Sentiment by topic
    st.header("4.2 Sentiment by Topic")

    topic_comparison = pd.crosstab(
        df["topic"],
        df["sentiment"]
    ).reindex(
        columns=["Positive", "Neutral", "Negative"],
        fill_value=0
    )

    st.dataframe(
        topic_comparison,
        width="stretch"
    )

    st.write(
        "The topic comparison helps identify the product features "
        "discussed in the comments and their associated sentiment labels. "
        "However, the number of comments for each topic is small and "
        "uneven. Therefore, the results are not sufficient to make a "
        "fair or conclusive comparison between topics."
    )

    # 4.3 Product team summary
    st.header("4.3 Product Team Summary")

    st.write(
        """
        The analysis of 18 synthetic customer reviews shows that positive
        feedback is the largest sentiment category, with 9 comments,
        followed by 6 negative and 3 neutral comments. Customers frequently
        discussed insulation, product usability, design, and quality.
        The word cloud highlights frequently used terms related to product
        features and customer experiences. However, some reviews contain
        mixed opinions. For example, one customer praised the bottle's
        ability to keep drinks cold but complained that its paint scratches
        easily.

        A key limitation is the small sample size and the use of synthetic
        practice comments, which may not represent actual customer
        experiences. Furthermore, VADER may misclassify comments because
        it does not always understand context or mixed opinions.

        The product team should prioritise improving product durability,
        lid usability, and cleaning convenience. For better insights,
        the company should collect a larger and more representative set
        of genuine customer reviews, including ratings and specific
        feedback about individual product features.
        """
    )

    st.info(
        "Note: The 18 comments used in this lab are synthetic practice "
        "data and should not be interpreted as genuine customer feedback."
    )


# =========================================================
# SIDEBAR FOOTER
# =========================================================

st.sidebar.divider()
st.sidebar.caption("Data Visualization | Lab 8")
st.sidebar.caption("Qualitative Data and Sentiment Analysis")
