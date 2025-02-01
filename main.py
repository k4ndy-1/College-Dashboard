import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Load data
file_path = "./colleges.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=['WEBSITE'])  # Remove empty column

def main():
    st.set_page_config(page_title="College Dashboard", layout="wide")
    st.title("🎓 College Dashboard")
    
    page = st.sidebar.selectbox("Select a Page", ["Main Dashboard", "Chi-Square Analysis", "Hierarchical Clustering"])
    
    if page == "Main Dashboard":
        show_main_dashboard()
    elif page == "Chi-Square Analysis":
        show_chi_square_analysis()
    elif page == "Hierarchical Clustering":
        show_hierarchical_clustering()

def show_main_dashboard():
    # Filters
    cities = df['CITY'].unique()
    courses = df['COURSE'].unique()
    city_filter = st.sidebar.multiselect("Select City", cities, default=cities)
    course_filter = st.sidebar.multiselect("Select Course", courses, default=courses)
    
    # Apply filters
    filtered_df = df[df['CITY'].isin(city_filter) & df['COURSE'].isin(course_filter)]
    
    # College count per city visualization
    city_counts = filtered_df['CITY'].value_counts().reset_index()
    city_counts.columns = ['City', 'Number of Colleges']
    fig = px.bar(city_counts, x='City', y='Number of Colleges', title="Colleges per City")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display filtered data
    st.dataframe(filtered_df, height=600)

def show_chi_square_analysis():
    st.subheader("Course Availability Across Cities")
    contingency_table = pd.crosstab(df['CITY'], df['COURSE'])
    plt.figure(figsize=(12, 6))
    sns.heatmap(contingency_table, cmap="coolwarm", linewidths=0.5, annot=False)
    plt.xlabel("Course")
    plt.ylabel("City")
    st.pyplot(plt)
    
    # Chi-Square Test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    st.subheader("Chi-Square Test Results")
    st.write("Case Study 1:")
    st.write("**Null Hypothesis (H₀):** There is no relationship between city and course availability.")
    st.write("**Alternative Hypothesis (H₁):** There is a significant relationship between city and course availability.")
    st.write(f"**Chi-Square Statistic:** {chi2_stat:.2f}")
    st.write(f"**Degrees of Freedom:** {dof}")
    st.write(f"**p-value:** {p_value:.5f}")
    
    if p_value < 0.05:
        st.write("**Conclusion:** Since the p-value is very small (< 0.05), we reject the null hypothesis. This means that course availability significantly depends on the city.")
    else:
        st.write("**Conclusion:** Since the p-value is greater than 0.05, we fail to reject the null hypothesis. This means there is no strong evidence that course availability depends on the city.")

def show_hierarchical_clustering():
    st.subheader("Hierarchical Clustering - Dendrogram")
    
    # Encoding categorical data (City and Course) into numerical values for clustering
    df_encoded = pd.get_dummies(df[['CITY', 'COURSE']])
    linked = linkage(df_encoded, method='ward')
    
    # Plot Dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=True)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Colleges")
    plt.ylabel("Distance")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
