import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Explicit imports from your inference layer
from inference.ai_extractor import extract_invoice_data
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# Page Configuration
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="🤖",
    layout="wide"
)

# Header Section
st.markdown("""
# Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging
""")

st.divider()

# Centralized Session State Initialization for data preservation
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "qty": 1200, "dollars": 18500.0, "flag_qty": 50, 
        "flag_dollars": 352.95, "freight": 1.73, "item_qty": 162, "item_dollars": 2476.0
    }

# ==========================================
# CENTRALIZED EMAIL SENDING FUNCTION
# ==========================================
def send_automated_email(sender_email, sender_password, recipient_email, subject, body, dataframe_subset, attachment_filename):
    """
    Helper function to package and send filtered dataframes via SMTP.
    If no background credentials are loaded, falls back gracefully to simulation mode.
    """
    if not sender_email or not sender_password:
        return "SIMULATION_SUCCESS"
        
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    csv_bytes = dataframe_subset.to_csv(index=False).encode('utf-8')
    
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(csv_bytes)
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {attachment_filename}")
    msg.attach(part)
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return "LIVE_SUCCESS"
    except Exception as e:
        return f"ERROR: {e}"


# ==========================================
# SIDEBAR NAVIGATION & EMAIL OPERATIONS
# ==========================================
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    ["Freight Cost Prediction", "Invoice Manual Approval Flag"]
)

st.sidebar.markdown("""
**Business Impact**
- Improved cost forecasting
- Reduced invoice fraud & anomalies
- Faster finance operations
""")

st.sidebar.write("---")

# 📧 Simple & Clean Operational Email Interface
st.sidebar.subheader("📧 Send Automated Email")

if st.session_state.processed_data is not None:
    df_active = st.session_state.processed_data
    
    # ⚙️ Hybrid Credentials Router Configuration Toggle
    use_custom_creds = st.sidebar.checkbox("⚙️ Use my own Gmail configs", value=False)
    
    if use_custom_creds:
        st.sidebar.caption("Provide your own Gmail credentials to send from your personal address.")
        user_sender = st.sidebar.text_input("Your Sender Gmail", placeholder="you@gmail.com", key="user_sender_input")
        user_pwd = st.sidebar.text_input("Your App Password", placeholder="xxxx xxxx xxxx xxxx", type="password", key="user_pwd_input")
    else:
        # 1. Instruction Box (Placed right above Demo Mode)
        st.sidebar.warning("💡 Click the checkbox above to use your own Gmail credentials instead.")
        
        # 2. Status Box
        st.sidebar.info("🤖 **Demo Mode: Active**\n\nRouting emails automatically via our pre-configured project mailbox.")
        
        # Securely fetch background dummy credentials
        user_sender = os.environ.get("DUMMY_SENDER_EMAIL")
        user_pwd = os.environ.get("DUMMY_APP_PASSWORD")

    st.sidebar.write("---")
    
    # Box 1: Approved for payment
    st.sidebar.markdown("### 1. Approved for payment")
    treasury_email = st.sidebar.text_input("Enter Treasury Email", placeholder="treasury@company.com", key="treasury_input")
    if st.sidebar.button("✉️ Send Approved Invoices"):
        if treasury_email:
            df_safe = df_active[df_active["AI_Risk_Assessment"] == "✅ SAFE (Auto-Approve)"].copy()
            if not df_safe.empty:
                if "AI_Risk_Assessment" in df_safe.columns:
                    df_safe = df_safe.drop(columns=["AI_Risk_Assessment"])
                
                email_body = "Team,\n\nPlease find attached the approved batch invoices verified by our machine learning pipeline. Proceed with scheduling payments.\n\nBest Regards,\nInvoice Intelligence Pipeline"
                
                result = send_automated_email(user_sender, user_pwd, treasury_email, "Approved Invoices for Processing", email_body, df_safe, "approved_payments.csv")
                
                if result == "LIVE_SUCCESS":
                    st.sidebar.success("🚀 Dispatched! Results sent successfully.")
                elif result == "SIMULATION_SUCCESS":
                    st.sidebar.warning("⚠️ Credentials not detected in system workspace. Running simulation mode instead.")
                else:
                    st.sidebar.error(f"Mail delivery failed: {result}")
            else:
                st.sidebar.warning("No auto-approved records found in this dataset.")
        else:
            st.sidebar.error("Please provide a recipient email.")

    st.sidebar.write("---")
    
    # Box 2: Need Manual Approval
    st.sidebar.markdown("### 2. Need Manual Approval")
    audit_email = st.sidebar.text_input("Enter Audit Team Email", placeholder="audit@company.com", key="audit_input")
    if st.sidebar.button("⚠️ Send Flagged Invoices"):
        if audit_email:
            df_flagged = df_active[df_active["AI_Risk_Assessment"] == "⚠️ MANUAL APPROVAL"].copy()
            if not df_flagged.empty:
                email_body = "Team,\n\nURGENT: Please verify the attached anomalies flagged by the risk model for potential compliance issues.\n\nBest Regards,\nInvoice Intelligence Pipeline"
                
                result = send_automated_email(user_sender, user_pwd, audit_email, "URGENT: Flagged Invoices for Audit Review", email_body, df_flagged, "manual_review_needed.csv")
                
                if result == "LIVE_SUCCESS":
                    st.sidebar.success("🚀 Escalated! High-risk rows sent to audit desk.")
                elif result == "SIMULATION_SUCCESS":
                    st.sidebar.warning("⚠️ Credentials not detected in system workspace. Running simulation mode instead.")
                else:
                    st.sidebar.error(f"Mail delivery failed: {result}")
            else:
                st.sidebar.success("Excellent! Zero risk anomalies detected.")
        else:
            st.sidebar.error("Please provide an auditor email.")
else:
    st.sidebar.info("💡 Upload and process a spreadsheet to activate the email tools.")


# ==========================================
# ⚡ AUTOMATED INTELLIGENT PROCESSING BLOCK
# ==========================================
st.subheader("⚡ Automated Intelligent Processing")
uploaded_file = st.file_uploader("Upload an Invoice Document (PDF, PNG, JPG) or Batch Spreadsheet (CSV, XLSX)", type=["pdf", "png", "jpg", "csv", "xlsx"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    
    # --- BATCH SPREADSHEET PATH ---
    if file_name.endswith('.csv') or file_name.endswith('.xlsx'):
        st.info("📊 Spreadsheet detected. Processing full batch predictions...")
        
        if file_name.endswith('.csv'):
            df_batch = pd.read_csv(uploaded_file)
        else:
            df_batch = pd.read_excel(uploaded_file)
            
        with st.spinner("Running batch machine learning models..."):
            try:
                # 1. Generate Freight Cost Predictions
                q_col = next((c for c in df_batch.columns if 'qty' in c.lower() or 'quantity' in c.lower()), 'Quantity')
                d_col = next((c for c in df_batch.columns if 'dollar' in c.lower()), 'Dollars')
                
                freight_input = {
                    "Quantity": df_batch[q_col].tolist(),
                    "Dollars": df_batch[d_col].tolist()
                }
                freight_preds = predict_freight_cost(freight_input)['Predicted_Freight']
                df_batch["Predicted_Freight_Cost"] = np.round(freight_preds, 2)
                
                # 2. Generate Manual Approval Flags
                flag_input = {
                    "invoice_quantity": df_batch.get("Quantity", df_batch.get("invoice_quantity", [0]*len(df_batch))).tolist(),
                    "invoice_dollars": df_batch.get("Invoice Dollars", df_batch.get("invoice_dollars", [0.0]*len(df_batch))).tolist(),
                    "Freight": df_batch.get("Freight Cost", df_batch.get("Freight", [0.0]*len(df_batch))).tolist(),
                    "total_item_quantity": df_batch.get("Total Item Quantity", df_batch.get("total_item_quantity", [0]*len(df_batch))).tolist(),
                    "total_item_dollars": df_batch.get("Total Item Dollars", df_batch.get("total_item_dollars", [0.0]*len(df_batch))).tolist()
                }
                flag_preds = predict_invoice_flag(flag_input)['Predicted_Flag']
                df_batch["AI_Risk_Assessment"] = ["⚠️ MANUAL APPROVAL" if f == 1 else "✅ SAFE (Auto-Approve)" for f in flag_preds]

                st.success(f"Successfully evaluated all {len(df_batch)} rows!")
                
                # Dynamic column arrangement logic based on sidebar option chosen
                if selected_model == "Freight Cost Prediction":
                    ordered_columns = [
                        "Quantity", "Invoice Dollars", "Freight Cost", 
                        "Total Item Quantity", "Total Item Dollars", "Predicted_Freight_Cost"
                    ]
                else:
                    ordered_columns = [
                        "Quantity", "Invoice Dollars", "Freight Cost", 
                        "Total Item Quantity", "Total Item Dollars", "Predicted_Freight_Cost", "AI_Risk_Assessment"
                    ]
                
                final_cols = [c for c in ordered_columns if c in df_batch.columns]
                df_batch_display = df_batch[final_cols]

                # Update persistent session state for the email operations tool to grab
                st.session_state.processed_data = df_batch

                # Show the dynamic clean data window
                st.dataframe(df_batch_display, use_container_width=True)
                
                # --- CONDITIONALLY SCOPED COUNTER PLACEMENT LOGIC ---
                if selected_model == "Invoice Manual Approval Flag" and "AI_Risk_Assessment" in df_batch.columns:
                    safe_count = int((df_batch["AI_Risk_Assessment"] == "✅ SAFE (Auto-Approve)").sum())
                    manual_count = int((df_batch["AI_Risk_Assessment"] == "⚠️ MANUAL APPROVAL").sum())
                    
                    metric_left, metric_mid, metric_right = st.columns([4, 4, 2])
                    with metric_right:
                        st.markdown(
                            f'<p style="font-size:14px; text-align:right; margin-top:-10px; opacity:0.9;">'
                            f'✅ {safe_count} &nbsp;&nbsp;&nbsp;&nbsp; ⚠️ {manual_count}'
                            f'</p>', 
                            unsafe_allow_html=True
                        )
                
                # Provide full operational download
                csv_output = df_batch_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Comprehensive Analytics Results CSV",
                    data=csv_output,
                    file_name="comprehensive_batch_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error executing batch generation: {e}. Check your spreadsheet column configurations.")
                
    # --- SINGLE DOCUMENT FILE PATH (AI VISION) ---
    else:
        with st.spinner("AI Agent is scanning document image metrics..."):
            extracted_metrics = extract_invoice_data(uploaded_file)
            if extracted_metrics:
                st.success("Single invoice text extracted successfully via AI Vision!")
                st.session_state.form_data["qty"] = int(extracted_metrics.get("invoice_quantity", 0))
                st.session_state.form_data["dollars"] = float(extracted_metrics.get("invoice_dollars", 0.0))
                st.session_state.form_data["flag_qty"] = int(extracted_metrics.get("invoice_quantity", 0))
                st.session_state.form_data["flag_dollars"] = float(extracted_metrics.get("invoice_dollars", 0.0))
                st.session_state.form_data["freight"] = float(extracted_metrics.get("Freight", 0.0))
                st.session_state.form_data["item_qty"] = int(extracted_metrics.get("total_item_quantity", 0))
                st.session_state.form_data["item_dollars"] = float(extracted_metrics.get("total_item_dollars", 0.0))

st.divider()

# Manual Input Forms below
if selected_model == "Freight Cost Prediction":
    st.subheader("Manual Override: Freight Cost Form")
    with st.form("freight_form"):
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input("Quantity", min_value=1, value=st.session_state.form_data["qty"])
        with col2:
            dollars = st.number_input("Invoice Dollars", min_value=1.0, value=st.session_state.form_data["dollars"])
        submit_freight = st.form_submit_button("Predict Single Freight Cost")
        
    if submit_freight:
        input_data = {"Quantity": [quantity], "Dollars": [dollars]}
        prediction = predict_freight_cost(input_data)['Predicted_Freight']
        st.metric(label="Estimated Freight Cost", value=f"${prediction[0]:,.2f}")

else:
    st.subheader("Manual Override: Invoice Manual Approval Prediction")
    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=st.session_state.form_data["flag_qty"])
            freight = st.number_input("Freight Cost", min_value=0.0, value=st.session_state.form_data["freight"])
        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=st.session_state.form_data["flag_dollars"])
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=st.session_state.form_data["item_qty"])
        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=st.session_state.form_data["item_dollars"])
        submit_flag = st.form_submit_button("Evaluate Single Invoice Risk")
        
    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity], "invoice_dollars": [invoice_dollars],
            "Freight": [freight], "total_item_quantity": [total_item_quantity], "total_item_dollars": [total_item_dollars]
        }
        flag_prediction = predict_invoice_flag(input_data)['Predicted_Flag']
        if bool(flag_prediction[0]):
            st.error("⚠️ Invoice requires **MANUAL APPROVAL**")
        else:
            st.success("✅ Invoice is **SAFE for Auto-Approval**")