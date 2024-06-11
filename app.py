import streamlit as st
import pandas as pd

# Title of the app
st.sidebar.title("Finance Data Analysis App")

# Main menu in the sidebar
menu = st.sidebar.radio("Main Menu", ["Daily Single Stock Analysis", "Stock Universe Analysis", 
                                      "Universe vs Single Stock", "Industry Stock Analysis", 
                                      "X-Factor Linear Model", "X-Factor Rolling Linear Model", 
                                      "Other Features"])

# Horizontal bar to separate sections
st.sidebar.markdown("---")

# Load the appropriate page based on user selection
if menu == "Daily Single Stock Analysis":
    import daily_single_stock_analysis
    daily_single_stock_analysis.show()
elif menu == "Stock Universe Analysis":
    import stock_universe_analysis
    stock_universe_analysis.show()
elif menu == "Universe vs Single Stock":
    import universe_single_stock
    universe_single_stock.show()
elif menu == "Industry Stock Analysis":
    import industry_stock_analysis
    industry_stock_analysis.show()
elif menu == "X-Factor Linear Model":
    import x_factor_linear_model
    x_factor_linear_model.show()
elif menu == "X-Factor Rolling Linear Model":
    import rolling_window_x_factor_linear_model
    rolling_window_x_factor_linear_model.show()
else:
    st.write("Select a feature from the main menu.")
