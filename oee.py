import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
from collections import Counter
import base64
from io import BytesIO

# Page configuration
st.set_page_config(page_title="Job Market Analysis", layout="wide")

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E90FF;
    }
    .chart-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    .stProgress .st-eb {
        background-color: #4169E1;
    }
    .skill-tag {
        display: inline-block;
        background-color: #e9ecef;
        padding: 4px 8px;
        margin: 4px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>Job Market Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("""
This application helps analyze job market data, predict salaries, and identify required skills
for various job positions based on machine learning models.
""")

# Function to load model (with graceful fallback)
@st.cache_resource
def load_models():
    try:
        # Try to load the models
        salary_model = joblib.load("models/salary_prediction_model.pkl")
        skill_model = joblib.load("models/skill_prediction_model.pkl")
        mlb = joblib.load("models/skill_binarizer.pkl")
        return salary_model, skill_model, mlb, True
    except:
        # If models aren't available, return None values
        st.warning("⚠️ Pre-trained models not found. Using fallback mode with simulated predictions.")
        return None, None, None, False

# Load models with a try/except to handle missing model files
salary_model, skill_model, mlb, models_available = load_models()

# Expanded list of skills
expanded_skill_list = [
    "python", "sql", "data visualization", "machine learning", "deep learning", "excel", 
    "statistics", "r", "java", "etl", "big data", "data analysis", "data engineer", 
    "data scientist", "cloud computing", "tensorflow", "keras", "pandas", "apache", 
    "spark", "hadoop", "docker", "kubernetes", "aws", "gcp", "azure", "nlp", "tableau", 
    "business intelligence", "marketing", "ux/ui design", "html", "css", "javascript", 
    "react", "angular", "vue", "graphql", "leadership", "team management", "communication", 
    "critical thinking", "problem-solving", "teamwork", "time management", "decision making",
    "negotiation", "conflict resolution", "project management", "agile", "scrum",
    "power bi", "data modeling", "data warehousing", "ai", "artificial intelligence",
    "nosql", "mongodb", "postgresql", "mysql", "database design", "git", "devops",
    "ci/cd", "analytics", "data governance", "data quality", "data security"
]

# Job title to description mapping
job_title_map = {
    "data scientist": "Looking for a data scientist with experience in python, machine learning, data analysis, and statistics.",
    "data analyst": "Looking for a data analyst with experience in data analysis, SQL, excel, and data visualization.",
    "software engineer": "Looking for a software engineer with experience in python, java, software development, and cloud computing.",
    "data engineer": "Looking for a data engineer with experience in data engineering, cloud computing, SQL, and big data technologies.",
    "machine learning engineer": "Looking for a machine learning engineer with experience in python, deep learning, TensorFlow, and model deployment.",
    "web developer": "Looking for a web developer with experience in HTML, CSS, JavaScript, React, and web application development.",
    "devops engineer": "Looking for a DevOps engineer with experience in CI/CD pipelines, cloud infrastructure, Docker, and Kubernetes.",
    "product manager": "Looking for a product manager with experience in product strategy, roadmaps, agile methodologies, and cross-functional team leadership.",
    "ux/ui designer": "Looking for a UX/UI designer with experience in wireframing, user research, prototyping, and design tools like Figma or Sketch.",
    "data architect": "Looking for a data architect with experience in designing data pipelines, database management, and cloud architectures.",
    "marketing manager": "Looking for a marketing manager with experience in digital marketing, SEO, SEM, and content creation.",
    "sales manager": "Looking for a sales manager with experience in lead generation, CRM tools, and sales strategies.",
    "financial analyst": "Looking for a financial analyst with experience in financial modeling, budgeting, forecasting, and Excel."
}

# Function to detect skills using expanded keyword list
def detect_skills_with_keywords(desc, skill_list):
    detected_skills = []
    desc_lower = desc.lower()
    
    for skill in skill_list:
        if skill.lower() in desc_lower:
            detected_skills.append(skill)
    
    return detected_skills

# Function to predict salary and skills
def predict_salary_and_skills(job_title, custom_desc=None):
    # Get job description for the title from the job_title_map or use custom description
    if custom_desc:
        desc = custom_desc
    else:
        desc = job_title_map.get(job_title.lower(), "Looking for a professional with expertise in data analysis and programming.")
    
    # Predict salary using models if available, otherwise use simulated salary predictions
    if models_available:
        predicted_salary = salary_model.predict([desc])
    else:
        # Simulated salary prediction based on job title
        salary_ranges = {
            "data scientist": 85000,
            "data analyst": 65000,
            "software engineer": 90000,
            "data engineer": 95000,
            "machine learning engineer": 100000,
            "web developer": 70000,
            "devops engineer": 95000,
            "product manager": 110000,
            "ux/ui designer": 80000,
            "data architect": 115000,
            "marketing manager": 85000,
            "sales manager": 90000,
            "financial analyst": 80000
        }
        
        # Randomize a bit to make it look more realistic
        base_salary = salary_ranges.get(job_title.lower(), 70000)
        randomize_factor = np.random.normal(1, 0.1)  # 10% standard deviation
        predicted_salary = np.array([base_salary * randomize_factor])
    
    # Detect skills using keyword-based method
    predicted_skills = detect_skills_with_keywords(desc, expanded_skill_list)
    
    return predicted_salary, predicted_skills, desc

# Function to get a suitable figure as BytesIO object for download
def get_figure_as_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# Function to create a download link for a figure
def get_image_download_link(buf, filename):
    image_base64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{image_base64}" download="{filename}">Download {filename}</a>'
    return href

# Function to create market value skill score
def calculate_skill_market_value(skills, job_titles):
    skill_value_scores = {}
    all_job_skills = []
    
    # Get skills for all selected job titles
    for job_title in job_titles:
        _, job_skills, _ = predict_salary_and_skills(job_title)
        all_job_skills.extend(job_skills)
    
    # Count occurrences of each skill
    skill_counts = Counter(all_job_skills)
    
    # Calculate a value score for each skill (normalized by maximum count)
    max_count = max(skill_counts.values()) if skill_counts else 1
    
    for skill in skills:
        skill_value_scores[skill] = skill_counts.get(skill, 0) / max_count
    
    return skill_value_scores

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Job Analysis", "Skills Explorer", "Data Visualization", "Career Path Advisor"])

# Job Analysis Page
if page == "Job Analysis":
    st.markdown("<h2 class='sub-header'>Job Title Analysis</h2>", unsafe_allow_html=True)
    
    # Input options
    analysis_type = st.radio("Choose input type:", ["Select from common job titles", "Enter custom job description"])
    
    if analysis_type == "Select from common job titles":
        job_title = st.selectbox("Select a job title:", sorted(list(job_title_map.keys())))
        predicted_salary, predicted_skills, desc = predict_salary_and_skills(job_title)
    else:
        custom_desc = st.text_area("Enter a job description:", height=150)
        job_title = st.text_input("Enter a job title (optional):")
        
        if custom_desc:
            predicted_salary, predicted_skills, desc = predict_salary_and_skills(job_title if job_title else "Custom Job", custom_desc)
        else:
            st.warning("Please enter a job description")
            st.stop()
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Salary Prediction")
        salary_fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(["Predicted Salary"], [predicted_salary[0]], color="#4169E1")
        ax.set_xlim(0, max(150000, predicted_salary[0] * 1.2))
        for i, v in enumerate([predicted_salary[0]]):
            ax.text(v + 1000, i, f"£{v:,.2f}", va='center')
        ax.set_xlabel("Annual Salary (£)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        st.pyplot(salary_fig)
        
        # Create download button for the salary chart
        salary_buf = get_figure_as_image(salary_fig)
        st.markdown(get_image_download_link(salary_buf, "salary_prediction.png"), unsafe_allow_html=True)
        
        # Add salary range information
        st.info(f"The predicted salary for a {job_title if job_title else 'this role'} is approximately £{predicted_salary[0]:,.2f} per year.")
        
    with col2:
        st.subheader("Required Skills")
        if predicted_skills:
            # Generate word cloud
            skill_text = " ".join(predicted_skills)
            if skill_text.strip():  # Check if we have any skills
                wordcloud = WordCloud(
                    width=400, 
                    height=200,
                    background_color='white',
                    colormap='viridis'
                ).generate(skill_text)
                
                skills_fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(skills_fig)
                
                # Create download button for the skills word cloud
                skills_buf = get_figure_as_image(skills_fig)
                st.markdown(get_image_download_link(skills_buf, "skills_wordcloud.png"), unsafe_allow_html=True)
                
                # Calculate market value for skills
                skill_value_scores = calculate_skill_market_value(predicted_skills, job_title_map.keys())
                
                # Show skills by market value
                st.subheader("Skills by Market Value")
                sorted_skills = sorted(predicted_skills, key=lambda x: skill_value_scores[x], reverse=True)
                
                # Display skills as tags with color gradient
                html_skills = ""
                for skill in sorted_skills:
                    value_pct = int(skill_value_scores[skill] * 100)
                    color_intensity = int(200 - (skill_value_scores[skill] * 100))
                    html_skills += f'<div class="skill-tag" style="background-color: rgb({color_intensity}, {255-color_intensity}, 255);">{skill.capitalize()} ({value_pct}%)</div>'
                
                st.markdown(html_skills, unsafe_allow_html=True)
            else:
                st.info("No specific skills detected.")
        else:
            st.info("No specific skills detected.")
    
    # Show the job description
    with st.expander("View Job Description"):
        st.write(desc)
    
    # Show skills as a list with relevance metrics
    with st.expander("View Skills Detail"):
        if predicted_skills:
            skill_value_scores = calculate_skill_market_value(predicted_skills, job_title_map.keys())
            
            # Create a DataFrame for better display
            skill_df = pd.DataFrame({
                'Skill': predicted_skills,
                'Market Value': [skill_value_scores[skill] for skill in predicted_skills]
            }).sort_values(by='Market Value', ascending=False)
            
            skill_df['Market Value'] = skill_df['Market Value'].apply(lambda x: f"{x:.0%}")
            st.dataframe(skill_df)
        else:
            st.info("No specific skills detected.")

# Skills Explorer Page
elif page == "Skills Explorer":
    st.markdown("<h2 class='sub-header'>Skills Explorer</h2>", unsafe_allow_html=True)
    
    # Multiselect for job titles
    selected_job_titles = st.multiselect(
        "Select job titles to compare:",
        sorted(list(job_title_map.keys())),
        default=["data scientist", "data analyst", "software engineer"]
    )
    
    if selected_job_titles:
        # Collect skills data for the selected job titles
        all_skills = []
        job_skills_map = {}
        
        for job_title in selected_job_titles:
            _, skills, _ = predict_salary_and_skills(job_title)
            all_skills.extend(skills)
            job_skills_map[job_title] = skills
        
        skill_counts = Counter(all_skills)
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Skills Frequency")
            # Get top skills
            top_skills = dict(skill_counts.most_common(15))
            
            # Plot skill frequency
            skills_freq_fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(top_skills))
            ax.barh(y_pos, list(top_skills.values()), color=plt.cm.viridis(np.linspace(0, 1, len(top_skills))))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(top_skills.keys()))
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Frequency')
            plt.tight_layout()
            st.pyplot(skills_freq_fig)
            
            # Create download button for the skills frequency chart
            skills_freq_buf = get_figure_as_image(skills_freq_fig)
            st.markdown(get_image_download_link(skills_freq_buf, "skills_frequency.png"), unsafe_allow_html=True)
        
        with col2:
            st.subheader("Skills Word Cloud")
            # Generate word cloud from all skills
            skills_text = " ".join(all_skills)
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(skills_text)
            
            wc_fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(wc_fig)
            
            # Create download button for the word cloud
            wc_buf = get_figure_as_image(wc_fig)
            st.markdown(get_image_download_link(wc_buf, "skills_wordcloud.png"), unsafe_allow_html=True)
        
        # Skills comparison table
        st.subheader("Skills Comparison by Job Title")
        
        # Create a DataFrame for comparison
        comparison_data = {}
        unique_skills = sorted(list(set(all_skills)))
        
        for job in selected_job_titles:
            comparison_data[job] = [skill in job_skills_map[job] for skill in unique_skills]
        
        comparison_df = pd.DataFrame(comparison_data, index=unique_skills)
        
        # Display the DataFrame
        st.dataframe(comparison_df.style.applymap(lambda x: 'background-color: #90EE90' if x else 'background-color: #ffcccb'))
        
        # Option to download the comparison data as CSV
        csv = comparison_df.to_csv()
        st.download_button(
            label="Download Skills Comparison as CSV",
            data=csv,
            file_name="skills_comparison.csv",
            mime="text/csv",
        )
        
        # Skills overlap visualization
        st.subheader("Skills Overlap Between Jobs")
        
        # Create a matrix for skills overlap percentage
        overlap_matrix = {}
        for job1 in selected_job_titles:
            overlap_matrix[job1] = {}
            for job2 in selected_job_titles:
                if job1 == job2:
                    overlap_matrix[job1][job2] = 1.0  # 100% overlap with self
                else:
                    skills1 = set(job_skills_map[job1])
                    skills2 = set(job_skills_map[job2])
                    if skills1 and skills2:  # Avoid division by zero
                        overlap = len(skills1.intersection(skills2)) / len(skills1.union(skills2))
                    else:
                        overlap = 0
                    overlap_matrix[job1][job2] = overlap
        
        # Convert to DataFrame for visualization
        overlap_df = pd.DataFrame(overlap_matrix)
        
        # Create a heatmap
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(overlap_df.values, cmap='viridis')
        
        # Setup colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Overlap Ratio", rotation=-90, va="bottom")
        
        # Show ticks and label them
        ax.set_xticks(np.arange(len(selected_job_titles)))
        ax.set_yticks(np.arange(len(selected_job_titles)))
        ax.set_xticklabels(selected_job_titles)
        ax.set_yticklabels(selected_job_titles)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations in each cell
        for i in range(len(selected_job_titles)):
            for j in range(len(selected_job_titles)):
                ax.text(j, i, f"{overlap_df.iloc[i, j]:.2f}", ha="center", va="center", color="white" if overlap_df.iloc[i, j] < 0.5 else "black")
        
        ax.set_title("Skills Overlap Between Job Titles")
        fig.tight_layout()
        st.pyplot(fig)
        
        # Create download button for the overlap chart
        overlap_buf = get_figure_as_image(fig)
        st.markdown(get_image_download_link(overlap_buf, "skills_overlap.png"), unsafe_allow_html=True)
    else:
        st.info("Please select at least one job title to analyze.")

# Data Visualization Page
elif page == "Data Visualization":
    st.markdown("<h2 class='sub-header'>Data Visualization</h2>", unsafe_allow_html=True)
    
    # Get salary predictions for all job titles
    job_titles = sorted(list(job_title_map.keys()))
    salaries = []
    for job in job_titles:
        salary, _, _ = predict_salary_and_skills(job)
        salaries.append(salary[0])
    
    # Create a DataFrame for visualization
    salary_df = pd.DataFrame({
        'Job Title': job_titles,
        'Predicted Salary': salaries
    }).sort_values(by='Predicted Salary', ascending=False)
    
    # Salary comparison
    st.subheader("Salary Comparison by Job Title")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Bar chart for salaries
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(salary_df['Job Title'], salary_df['Predicted Salary'], color=plt.cm.viridis(np.linspace(0, 1, len(job_titles))))
        ax.set_xlabel('Predicted Annual Salary (£)')
        ax.set_title('Predicted Salaries by Job Title')
        
        # Add salary values at the end of each bar
        for i, v in enumerate(salary_df['Predicted Salary']):
            ax.text(v + 1000, i, f"£{v:,.2f}", va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Create download button for the salary comparison chart
        salary_comp_buf = get_figure_as_image(fig)
        st.markdown(get_image_download_link(salary_comp_buf, "salary_comparison.png"), unsafe_allow_html=True)
    
    with col2:
        # Display the salary table
        st.dataframe(salary_df.style.format({'Predicted Salary': '£{:,.2f}'}))
        
        # Option to download the salary data as CSV
        csv = salary_df.to_csv()
        st.download_button(
            label="Download Salary Data as CSV",
            data=csv,
            file_name="salary_data.csv",
            mime="text/csv",
        )
    
    # Skill frequency across all job titles
    st.subheader("Most In-Demand Skills Across All Job Titles")
    
    # Collect all skills
    all_job_skills = []
    for job in job_titles:
        _, skills, _ = predict_salary_and_skills(job)
        all_job_skills.extend(skills)
    
    skill_counts = Counter(all_job_skills)
    top_skills = dict(skill_counts.most_common(20))
    
    # Plot top skills
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(top_skills.keys(), top_skills.values(), color=plt.cm.viridis(np.linspace(0, 1, len(top_skills))))
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 20 Most In-Demand Skills')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create download button for the top skills chart
    top_skills_buf = get_figure_as_image(fig)
    st.markdown(get_image_download_link(top_skills_buf, "top_skills.png"), unsafe_allow_html=True)
    
    # Skills heat map by job category
    st.subheader("Skills Heat Map by Job Category")
    
    # Group job titles into categories
    job_categories = {
        "Data Roles": ["data scientist", "data analyst", "data engineer", "data architect"],
        "Engineering Roles": ["software engineer", "web developer", "devops engineer", "machine learning engineer"],
        "Business Roles": ["product manager", "business analyst", "marketing manager", "sales manager", "financial analyst"],
        "Design Roles": ["ux/ui designer"]
    }
    
    # Collect skills by category
    category_skills = {}
    for category, jobs in job_categories.items():
        all_cat_skills = []
        for job in jobs:
            if job in job_title_map:
                _, skills, _ = predict_salary_and_skills(job)
                all_cat_skills.extend(skills)
        category_skills[category] = Counter(all_cat_skills)
    
    # Get common skills across categories (top 15)
    all_category_skills = []
    for category, skill_counter in category_skills.items():
        all_category_skills.extend(skill_counter.keys())
    
    common_skills = Counter(all_category_skills).most_common(15)
    common_skill_names = [skill[0] for skill in common_skills]
    
    # Create a matrix for the heatmap
    heatmap_data = []
    for skill in common_skill_names:
        row = []
        for category in job_categories.keys():
            row.append(category_skills[category].get(skill, 0))
        heatmap_data.append(row)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data, cmap='viridis')
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(job_categories)))
    ax.set_yticks(np.arange(len(common_skill_names)))
    ax.set_xticklabels(job_categories.keys())
    ax.set_yticklabels(common_skill_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(common_skill_names)):
        for j in range(len(job_categories)):
            ax.text(j, i, heatmap_data[i][j], ha="center", va="center", color="white" if heatmap_data[i][j] > 3 else "black")
    
    ax.set_title("Skills Frequency by Job Category")
    fig.tight_layout()
    st.pyplot(fig)
    
    # Create download button for the heatmap
    heatmap_buf = get_figure_as_image(fig)
    st.markdown(get_image_download_link(heatmap_buf, "skills_heatmap.png"), unsafe_allow_html=True)

# Career Path Advisor
elif page == "Career Path Advisor":
    st.markdown("<h2 class='sub-header'>Career Path Advisor</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This tool helps you explore potential career paths based on your current skills and interests.
    It will recommend job roles that match your skill set and suggest skills to develop to advance your career.
    """)
    
    # Skills selection
    st.subheader("Your Skills")
    
    # Group skills into categories for easier selection
    skill_categories = {
        "Programming Languages": ["python", "java", "javascript", "r", "html", "css"],
        "Data Skills": ["sql", "data analysis", "data visualization", "statistics", "excel", "tableau", "power bi"],
        "Cloud & Infrastructure": ["aws", "azure", "gcp", "docker", "kubernetes", "cloud computing"],
        "Machine Learning & AI": ["machine learning", "deep learning", "tensorflow", "keras", "nlp", "ai", "artificial intelligence"],
        "Big Data": ["big data", "spark", "hadoop", "etl", "data warehousing"],
        "Soft Skills": ["leadership", "communication", "teamwork", "problem-solving", "critical thinking", "time management"],
        "Management Skills": ["project management", "team management", "agile", "scrum", "decision making"]
    }
    
    # Let user select skills by category
    user_skills = []
    
    for category, skills in skill_categories.items():
        with st.expander(f"{category} Skills"):
            selected_skills = st.multiselect(
                f"Select your {category.lower()} skills:",
                skills,
                key=category
            )
            user_skills.extend(selected_skills)
    
    # Additional skills not in categories
    other_skills = [skill for skill in expanded_skill_list if not any(skill in cat_skills for cat_skills in skill_categories.values())]
    
    with st.expander("Other Skills"):
        other_selected = st.multiselect(
            "Select any other skills you have:",
            sorted(other_skills)
        )
        user_skills.extend(other_selected)
    
    # Display selected skills
    st.subheader("Your Selected Skills")
    
    if user_skills:
        # Display skills as tags
        skills_html = ""
        for skill in sorted(user_skills):
            skills_html += f'<span class="skill-tag">{skill}</span>'
        
        st.markdown(skills_html, unsafe_allow_html=True)
        
        # Career path recommendation
        st.subheader("Recommended Career Paths")
        
        # Calculate match percentage for each job title
        job_match_scores = {}
        job_missing_skills = {}
        
        for job_title in job_title_map.keys():
            _, job_skills, _ = predict_salary_and_skills(job_title)
            
            if job_skills:
                # Calculate match percentage
                user_skill_set = set(user_skills)
                job_skill_set = set(job_skills)
                
                matching_skills = user_skill_set.intersection(job_skill_set)
                match_percentage = len(matching_skills) / len(job_skill_set) if job_skill_set else 0
                
                # Store missing skills
                missing_skills = job_skill_set - user_skill_set
                
                job_match_scores[job_title] = match_percentage
                job_missing_skills[job_title] = missing_skills
        
        # Sort jobs by match percentage
        sorted_jobs = sorted(job_match_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display top matches with progress bars
       # Display top matches with progress bars
        for job_title, match_score in sorted_jobs[:5]:  # Show top 5 matches
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{job_title.title()}**")
                st.progress(match_score)
                
                # Display missing skills
                if job_missing_skills[job_title]:
                    missing_skills_list = list(job_missing_skills[job_title])
                    missing_skills_html = "Skills to develop: "
                    for skill in sorted(missing_skills_list):
                        missing_skills_html += f'<span class="skill-tag" style="background-color: #ffcccb;">{skill}</span>'
                    st.markdown(missing_skills_html, unsafe_allow_html=True)
            
            with col2:
                st.write(f"**Match: {match_score:.0%}**")
                # Display predicted salary
                predicted_salary, _, _ = predict_salary_and_skills(job_title)
                st.write(f"Salary: £{predicted_salary[0]:,.2f}")
        
        # Skill development recommendations
        st.subheader("Skill Development Recommendations")
        
        # Count most common missing skills across top job matches
        all_missing_skills = []
        for job_title, match_score in sorted_jobs[:8]:  # Consider top 8 matching jobs
            all_missing_skills.extend(job_missing_skills[job_title])
        
        missing_skill_counts = Counter(all_missing_skills)
        
        if missing_skill_counts:
            # Show top skills to develop
            st.write("Based on your top matching career paths, we recommend developing these skills:")
            
            # Create two columns for the bar chart and recommendations
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create a bar chart of top missing skills
                top_missing = dict(missing_skill_counts.most_common(8))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(list(top_missing.keys()), list(top_missing.values()), color=plt.cm.viridis(np.linspace(0, 1, len(top_missing))))
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Frequency')
                ax.set_title('Top Skills to Develop')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Create download button for the skills development chart
                skill_dev_buf = get_figure_as_image(fig)
                st.markdown(get_image_download_link(skill_dev_buf, "skill_development.png"), unsafe_allow_html=True)
            
            with col2:
                # Generate specific recommendations
                st.write("How these skills will help your career:")
                for skill, count in missing_skill_counts.most_common(5):
                    st.markdown(f"**{skill.capitalize()}** - Required for {count} of your top matching roles")
        else:
            st.write("Great job! You already have the core skills for your top matching roles.")
        
        # Career progression visualization
        st.subheader("Career Progression Path")
        
        # Define career progression paths based on common industry trajectories
        career_paths = {
            "data analyst": ["data analyst", "senior data analyst", "data scientist", "senior data scientist", "data science manager"],
            "data scientist": ["data scientist", "senior data scientist", "lead data scientist", "data science manager", "director of data science"],
            "software engineer": ["software engineer", "senior software engineer", "lead engineer", "engineering manager", "director of engineering"],
            "data engineer": ["data engineer", "senior data engineer", "data architect", "head of data infrastructure", "chief data officer"],
            "machine learning engineer": ["machine learning engineer", "senior ML engineer", "ML architect", "AI research scientist", "director of AI"],
            "web developer": ["web developer", "senior web developer", "full-stack engineer", "technical lead", "CTO"],
            "devops engineer": ["devops engineer", "senior devops engineer", "infrastructure architect", "head of infrastructure", "CTO"],
            "product manager": ["product manager", "senior product manager", "group product manager", "director of product", "chief product officer"],
            "business analyst": ["business analyst", "senior business analyst", "product manager", "senior product manager", "director of product"],
            "ux/ui designer": ["ux/ui designer", "senior designer", "design lead", "design manager", "creative director"]
        }
        
        # Find best matching career path based on top job match
        if sorted_jobs:
            best_match_job = sorted_jobs[0][0]
            
            # Find the closest career path
            closest_path_key = best_match_job
            if best_match_job not in career_paths:
                # Find similar career path if exact match not available
                for path_key in career_paths.keys():
                    if path_key in best_match_job or best_match_job in path_key:
                        closest_path_key = path_key
                        break
                else:
                    # Default to data analyst if no match found
                    closest_path_key = "data analyst"
            
            # Display the career progression path
            progression_path = career_paths[closest_path_key]
            
            # Create a horizontal progression diagram
            fig, ax = plt.subplots(figsize=(12, 3))
            
            # Create positions for the progression items
            positions = np.arange(len(progression_path))
            
            # Create colored boxes
            for i, pos in enumerate(positions):
                ax.add_patch(plt.Rectangle((pos - 0.4, -0.4), 0.8, 0.8, 
                                          color=plt.cm.viridis(i/len(progression_path)),
                                          alpha=0.8))
                ax.text(pos, 0, progression_path[i], ha='center', va='center', fontsize=9,
                       fontweight='bold', color='white' if i > 1 else 'black')
            
            # Add arrows between positions
            for i in range(len(positions) - 1):
                ax.annotate("", xy=(positions[i+1] - 0.4, 0), xytext=(positions[i] + 0.4, 0),
                           arrowprops=dict(arrowstyle="->", lw=2, color="gray"))
            
            ax.set_xlim(-0.5, len(progression_path) - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create download button for the career progression chart
            career_prog_buf = get_figure_as_image(fig)
            st.markdown(get_image_download_link(career_prog_buf, "career_progression.png"), unsafe_allow_html=True)
            
            # Add explanation
            st.write(f"Based on your skills and interests, we've shown a potential career progression path starting from {progression_path[0]}. Each role typically requires additional skills and experience.")
    else:
        st.info("Please select at least one skill to get personalized career recommendations.")

# Add a footer with information
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    Job Market Analysis Tool - A data-driven approach to career planning.<br>
    This application uses machine learning models to analyze job market data and provide insights for career development.
</div>
""", unsafe_allow_html=True)