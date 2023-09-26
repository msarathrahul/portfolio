import streamlit as st
#from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="Sarath Rahul Malla Portfolio")

# Initializing two columns
col1, col2 = st.columns(2)

# Load profile image
#image = Image.open('image.PNG')

# Display profile image in the first column
#with col1:
    #st.image(image, width=250)

# Display name and contact information in the second column
with col1:
    # Page title with Markdown styling
    st.markdown("# Sarath Rahul Malla")
    
    # Contact section
    st.write("📧 **Contact Me**")
    st.write("Phone: +91 7799889953")
    st.write("Email: msarathrahul@gmail.com")
    st.write("LinkedIn: [linkedin.com/msarathrahul](https://www.linkedin.com/in/msarathrahul/)")
    st.write("GitHub: [github.com/msarathrahul](https://github.com/msarathrahul)")

# About Me section
st.markdown("## About Me :wave:")
st.write("Interned as **Junior Data Scientist** at The University of Edinburgh, driven to build reliable models for effective Decision making. Strong expertise in Data preprocessing, Feature engineering, and Data visualization. Adept at translating data-driven insights into actionable recommendations. 💼")

st.markdown("---")

st.markdown("## Check sidebar(On left) for mare details")

# Sidebar for navigation
st.sidebar.title("Portfolio Navigation")
selected_section = st.sidebar.radio("Go to", ["Experience :briefcase:", "Data Science Projects :rocket:", "Technical Skills :computer:", "Education :mortar_board:", "Achievements and Awards :star:", "Contact Me :email:"])

# Experience section
if selected_section == "Experience :briefcase:":
    st.markdown("## Experience :briefcase:")

    # University of Edinburgh Internship
    st.markdown("### Junior Data Scientist - DSU (Intern) - University of Edinburgh")
    st.write("**June 2023 - July 2023**")
    st.write("**Edinburgh, UK**")
    st.write("I interned as a Junior Data Scientist at the University of Edinburgh, where I worked on the following projects:")
    st.write("1. **Machine Learning - Realistic Material Simulation**")
    st.write("Developed a Graph Network-based Simulators (GNS) framework, inspired by Deep Mind’s 'Learning to Simulate' project, to simulate complex physical domains in realistic environments. 🧪")
    st.write("- Demonstrated outstanding generalization, achieving 94.92% accuracy in predicting material behavior across diverse initial conditions, extended timeframes, and simulations with 10 times more particles.")
    st.write("- Enhanced long-term performance by optimizing the number of message-passing steps, reducing simulation error by 36%, and introducing noise during training for a 2.4% improvement in stability.")
    st.write("2. **Optimization - Multi Objective Bayesian optimization**")
    st.write("Optimized two crucial variables, lambda r and lambda b, in Raman Spectroscopy to enhance the understanding of Liver functionality, significantly improving simulation accuracy and efficiency. The conventional naive method necessitated exhaustive script runs, consuming considerable time and computational resources. 📈")
    st.write("- Implemented Multi-Objective Bayesian Optimization, resulting in a remarkable 62% reduction in runtime compared to the traditional naive approaches.")
    st.write("- The optimization process resulted in a remarkable 42% reduction in differential cost, reflecting heightened accuracy, and a concurrent 28% reduction in regularization cost, crucial for managing model complexity and preventing over-fitting.")
    st.write("- Git was employed for version control, enabling organized code management, collaborative development, and efficient code pushing, which contributed to project success.")

    # University of Edinburgh
    st.markdown("### Tutor and Demonstrators - University of Edinburgh")
    st.write("**Oct 2023 - Aug 2023**")
    st.write("**Edinburgh, UK**")

    # Cognizant Technology Solutions
    st.markdown("### Programmer Analyst - Cognizant Technology Solutions")
    st.write("**Feb 2021 - Aug 2022**")
    st.write("**Chennai, India**")
    st.write("During my time at Cognizant, I worked on the following tasks:")
    st.write("- Implemented performance tuning techniques such as indexing and query optimization to improve speed and efficiency of SQL queries, resulting in 30% reduction in query execution time.")
    st.write("- Ensured proper functionality of automated nightly jobs through effective utilization of Control-M application.")
    st.write("- Collaborated with a team of 10 members to successfully transition a company from On-Premise to Cloud-based infrastructure, resulting in 20% reduction in infrastructure costs and improved system performance.")
    st.write("- Served as BCM Coordinator, managing communication and coordination efforts for a team of 15+ members during various business continuity events.")
    st.write("- Created 50+ mappings to facilitate the seamless flow of data between Essbase cubes, resulting in improved data accuracy and reporting capabilities.")
    st.write("- Led the loading of metadata and data validations for multiple projects, resulting in the successful deployment of 10+ enterprise-level applications with no data-related issues reported.")

# Data Science Projects section
elif selected_section == "Data Science Projects :rocket:":
    st.markdown("## Data Science Projects :rocket:")

    #ECG Abnormality Detection Using LSTM Neural Network
    st.markdown("### Electrocardiogram(ECG) Abnormality Detection Using LSTM Neural Network")
    st.write("Spearheaded the development and deployment of an LSTM (Long Short-Term Memory) neural network to detect cardiac abnormalities in ECG signals. [Click here](https://github.com/msarathrahul/anomaly_detection_ECG) 🩺")
    st.write("- Achieved an impressive accuracy of 97.8% in detecting ECG abnormalities, surpassing the industry-standard mean absolute loss threshold of 0.26, where anything above this threshold is considered abnormal.")
    st.write("- Expertly implemented data preprocessing techniques to enhance the quality of ECG signal data, ensuring precise anomaly detection.")
    st.write("- Conducted meticulous hyperparameter tuning to optimize the LSTM model, resulting in exceptional generalization capabilities.")

    #Computer Vision for Malaria Detection
    st.markdown("### Computer Vision for Malaria Detection")
    st.write("Led the end-to-end development and deployment of a robust deep learning-based computer vision model using the tiny-VGG architecture for malaria detection, achieving an outstanding accuracy of 94.97% and an impressive F1 score of 1.0 on the test data. [Click here](https://github.com/msarathrahul/stroke_prediction) 🦟")
    st.write("- Implemented transfer learning techniques to fine-tune the tiny-VGG architecture, effectively leveraging features learned from similar image recognition tasks.")
    st.write("- Independently curated and annotated a diverse dataset of 25,000 cell images, achieving a balanced trade-off between sensitivity and specificity in the model's predictions for malaria detection.")
    st.write("- Leveraged the power of Torch and CUDA to optimize model performance, significantly reducing computation time by 40% during inference.")

    # Heart Stroke Prediction Classifier
    st.markdown("### Heart Stroke Prediction Classifier")
    st.write("Spearheaded the development of a groundbreaking heart stroke prediction classifier, achieving an exceptional accuracy of 100% (Cross-validation) and an impressively low False Negative rate of 0 on test data. [Click here](https://github.com/msarathrahul/stroke_prediction) ❤️")
    st.write("- Leveraged cutting-edge technologies and algorithms, including Ensemble techniques, Bagging, XGBoost, and Random Forest Classifier, to create a robust and high-performing model.")
    st.write("- Utilized state-of-the-art data preprocessing techniques to handle complex medical data, ensuring seamless integration with the model and reducing data noise by 30%.")
    st.write("- Processed a large-scale dataset of over 9,000 patient records, demonstrating strong data handling and management skills.")

    # Churn Prediction Model
    st.markdown("### Churn Prediction Model in Telecom Industry")
    st.write("Developed a Machine learning model for churn prediction in the telecom industry, achieving a remarkable accuracy rate of **89.22%**. 📱")
    st.write("- Employed advanced Data analysis and feature engineering to identify key factors influencing customer churn.")
    st.write("- Utilized data pipelines and employed a variety of advanced machine learning techniques such as Bagging, Boosting, and Stacking, while also fine-tuning hyperparameters to optimize predictive performance.")
    st.write("- Conducted rigorous model validation and evaluation, including cross-validation and ROC curve analysis, ensuring robust predictive capabilities.")

    # Time Series Forecasting
    st.markdown("### Time Series Forecasting for Energy Consumption")
    st.write("Developed a robust time series forecasting solution to predict future energy consumption patterns, optimizing resource allocation and cost management in the energy sector. ⚡")
    st.write("- Formulated a Time Series Forecasting (TSF) framework, drawing inspiration from state-of-the-art research in time series prediction.")
    st.write("- Demonstrated remarkable predictive capability, achieving an accuracy rate of 92.7% in forecasting energy consumption across diverse scenarios, including peak demand periods and seasonal fluctuations.")
    st.write("- Improved model stability and long-term performance by optimizing hyperparameters, reducing prediction errors by 34%, and introducing adaptive learning rate schedules.")

    # Video Recommendation System
    st.markdown("### Dynamic Video Recommendation System for Web Content using PySpark")
    st.write("Developed a dynamic video recommendation system leveraging Natural Language Processing (NLP) techniques to enhance user engagement on a web-based video content platform. 🎥")
    st.write("- Designed and implemented web scraping automation to refresh video data twice in a month, ensuring real-time content updates.")
    st.write("- Utilized NLP algorithms for semantic analysis of video titles and tags, enabling personalized recommendations.")
    st.write("- Incorporated user-friendly custom filtering options, allowing users to tailor content preferences based on specific tags or words.")

    # Financial Fraud Anomaly Detection
    st.markdown("### Financial Fraud Anomaly Detection")
    st.write("Developed an anomaly detection system using Auto Encoders that fortified financial transactions against fraud, yielding substantial improvements in security and operational efficiency. 💳")
    st.write("- Cleaned and preprocessed transaction data, addressing 98% of missing values and outliers to ensure accurate anomaly identification.")
    st.write("- Engineered 15 new features, leading to a 30% increase in the model's ability to capture nuanced anomalies")
    st.write("- Fine-tuned threshold values, optimizing the model for an 85% precision rate and capturing 93% of genuine fraud cases.")

    # Sentiment Analysis of Customer Reviews using BERT
    st.markdown("### Sentiment Analysis of Customer Reviews using BERT (Large Language Model)")
    st.write("Employed advanced NLP, conducted sentiment analysis on e-commerce customer reviews by scraping the reviews from website. Harnessing BERT, achieved 94% (off by 1) accuracy in sentiment classification. 🛒")
    st.write("- Preprocessed and tokenized text, accommodating user-generated content, special characters and emojis.")
    st.write("- Fine-tuned BERT on a custom dataset to capture nuanced sentiment across diverse feedback.")
    st.write("- Enhanced performance with attention mechanisms and transfer learning.")

# Technical Skills section
elif selected_section == "Technical Skills :computer:":
    st.markdown("## Technical Skills :computer:")
    st.write("🛠️ **Machine learning libraries** : Numpy, Pandas, Scikit-Learn, PyTorch, SciPy, NLTK, Hugging Face, SpaCy")
    st.write("💾 **Big Data tools** : PySpark, Hadoop (HDFS), MapReduce")
    st.write("💻 **Programming languages** : Python, SQL (MySQL & Postgres), MATLAB, R, C, Java")
    st.write("📊 **Data visualization and tools** : Tableau, PowerBI, Jupyter Notebook, Excel, Git, Docker, Oracle Essbase")
    st.write("☁️ **Cloud platforms** : Amazon Web Services (AWS), Google Cloud Platform (GCP)")
    st.write("🕸️ **Web scraping packages** : Requests, Selenium, BeautifulSoup")

# Education section
elif selected_section == "Education :mortar_board:":
    st.markdown("## Education :mortar_board:")

    # MSc in Data Science
    st.markdown("### MSc in Data Science - University of Edinburgh")
    st.write("Expected 2023")
    st.write("Modules: Applied Machine Learning, Database Systems, Natural Language Processing, and Incomplete Data Analysis")

    # B.Tech in Mechanical Engineering
    st.markdown("### B.Tech in Mechanical Engineering - ANITS")
    st.write("2016 - 2020")
    st.write("CGPA: 8.36/10")
    st.write("Publication: Optimization of Cutting Parameters in CNC Turning Machine using ANOVA technique in TAGUCHI method")

# Achievements and Awards section
elif selected_section == "Achievements and Awards :star:":
    st.markdown("## Achievements and Awards :star:")
    st.write("- Passed Python, Machine learning, MySQL, and Microsoft Excel LinkedIn skill assessments.")
    st.write("- Certificate in Gesture Controlled Maze Workshop, Shaastra 2019, IIT Madras")
    st.write("- APSSDC Certified for successful completion of Problem Solving Skills using C workshop, 2018")
    st.write("- Certificate of Merit in NSTSE and SLSTSE Maths Olympiads, 2013 and 2012")
    st.write("- Junior Level Screening Test Certified, The Association of Mathematics Teacher of India, 2012")
    st.write("- International Rank 81 in International Mathematics Olympiad conducted by SOF, 2012")

# Contact Me section
elif selected_section == "Contact Me :email:":
    st.markdown("## Contact Me :email:")
    st.write("Phone: +91 7799889953")
    st.write("📧 Email: msarathrahul@gmail.com")
    st.write("LinkedIn: [linkedin.com/msarathrahul](https://www.linkedin.com/in/msarathrahul/)")
    st.write("GitHub: [github.com/msarathrahul](https://github.com/msarathrahul)")
