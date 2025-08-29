import streamlit as st
import tempfile
import os
from pathlib import Path
import pandas as pd
import pipeline

st.set_page_config(
    page_title="Deepfake Intelligence Radar — Image Triage",
    page_icon="🛰️",
    layout="wide",
)

st.title("Deepfake Intelligence Radar — Image Triage")

# Add tech stack badges
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("Upload images to analyze for potential manipulation, deepfakes, or AI-generation.")
with col2:
    st.markdown("**🚀 Built with:**")
    st.markdown("[![Daft](https://img.shields.io/badge/Daft-DataFrame_Engine-FF6B6B?style=for-the-badge)](https://www.getdaft.io/)")
with col3:
    st.markdown("**☁️ Powered by:**")
    st.markdown("[![Daytona](https://img.shields.io/badge/Daytona-Dev_Platform-4A90E2?style=for-the-badge)](https://daytona.io/)")

# Add some spacing
st.divider()

# File uploader section
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Choose image files to analyze",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Select one or more JPG/PNG images for forensic analysis",
        key="image_uploader",
    )

with col2:
    st.write("")  # Add spacing
    st.write("")  # Add spacing
    analyze_button = st.button(
        "🔍 Analyze",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
        help="Click to start forensic analysis" if uploaded_files else "Please upload images first",
    )

# Display upload status
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded and ready for analysis")
    
    # Show file details in an expander
    with st.expander("View uploaded files", expanded=False):
        for i, file in enumerate(uploaded_files, 1):
            file_size_mb = file.size / (1024 * 1024)
            st.text(f"{i}. {file.name} ({file_size_mb:.2f} MB)")
else:
    st.info("👆 Please upload one or more image files to begin")

# Process when analyze button is clicked
if analyze_button and uploaded_files:
    st.divider()
    st.header("📊 Analysis Results")
    
    with st.spinner(f"Analyzing {len(uploaded_files)} image(s)..."):
        # Create temporary directory for saving files
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = []
            
            # Save uploaded files to temp directory
            for uploaded_file in uploaded_files:
                # Create safe filename
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Write file to disk
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                saved_paths.append(temp_path)
            
            try:
                # Build Daft DataFrame and process images
                df = pipeline.build_df(saved_paths)
                
                # Collect results (execute the lazy dataframe)
                results = df.collect()
                
                # Convert to pandas DataFrame for display
                # Extract only the columns we want to show
                display_data = []
                for row in results:
                    display_data.append({
                        "File": Path(row["path"]).name,  # Just filename, not full path
                        "ELA Score": f"{row['ela_score']:.4f}",
                        "FFT Score": f"{row['fft_score']:.4f}",
                        "Face Detected": "✅ Yes" if row["face_conf"] > 0.5 else "❌ No",
                        "Suspicion Score": f"{row['suspicion']:.4f}",
                    })
                
                results_df = pd.DataFrame(display_data)
                
                # Display summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_suspicion = sum(float(d["Suspicion Score"]) for d in display_data) / len(display_data)
                    st.metric("Average Suspicion", f"{avg_suspicion:.3f}")
                
                with col2:
                    high_risk = sum(1 for d in display_data if float(d["Suspicion Score"]) > 0.7)
                    st.metric("High Risk Images", high_risk)
                
                with col3:
                    faces_detected = sum(1 for d in display_data if "✅" in d["Face Detected"])
                    st.metric("Faces Detected", f"{faces_detected}/{len(display_data)}")
                
                with col4:
                    max_suspicion = max(float(d["Suspicion Score"]) for d in display_data)
                    st.metric("Max Suspicion", f"{max_suspicion:.3f}")
                
                st.divider()
                
                # Display the results table
                st.subheader("🔍 Detailed Analysis Results")
                
                # Style the dataframe
                def color_suspicion(val):
                    """Color code suspicion scores"""
                    if isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit():
                        score = float(val)
                        if score < 0.3:
                            color = 'background-color: #90EE90'  # Light green
                        elif score < 0.6:
                            color = 'background-color: #FFD700'  # Gold
                        elif score < 0.8:
                            color = 'background-color: #FFA500'  # Orange
                        else:
                            color = 'background-color: #FF6B6B'  # Light red
                        return color
                    return ''
                
                # Apply styling to suspicion column
                styled_df = results_df.style.applymap(
                    color_suspicion,
                    subset=['Suspicion Score']
                )
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                )
                
                # Add download button for CSV export
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"image_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the analysis results as a CSV file for further processing",
                    key="download_csv",
                )
                
                # Add interpretation guide
                with st.expander("📖 How to interpret the scores", expanded=False):
                    st.markdown("""
                    **Suspicion Score Ranges:**
                    - 🟢 **0.0 - 0.3**: Low suspicion (likely authentic)
                    - 🟡 **0.3 - 0.6**: Moderate suspicion (possible manipulation)
                    - 🟠 **0.6 - 0.8**: High suspicion (likely manipulated)
                    - 🔴 **0.8 - 1.0**: Very high suspicion (strong evidence of manipulation)
                    
                    **Individual Scores:**
                    - **ELA Score**: Higher values indicate compression artifacts or edits
                    - **FFT Score**: Higher values suggest artificial patterns or noise
                    - **Face Detected**: Presence of faces increases deepfake concern
                    """)
                
                # Store results in session state for potential export
                st.session_state['analysis_results'] = results_df
                
                # Display ELA heatmaps
                st.divider()
                st.subheader("🔥 ELA Heatmap Analysis")
                st.markdown("Error Level Analysis visualizations showing potential manipulation areas")
                
                # Determine number of columns based on image count
                num_images = len(results)
                if num_images == 1:
                    cols_per_row = 1
                elif num_images == 2:
                    cols_per_row = 2
                else:
                    cols_per_row = 3
                
                # Create rows of heatmaps
                for i in range(0, num_images, cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < num_images:
                            row = results[idx]
                            
                            with cols[j]:
                                # Get filename and suspicion score
                                filename = Path(row["path"]).name
                                suspicion = row["suspicion"]
                                
                                # Determine risk level for color coding
                                if suspicion < 0.3:
                                    risk_emoji = "🟢"
                                    risk_text = "Low Risk"
                                elif suspicion < 0.6:
                                    risk_emoji = "🟡"
                                    risk_text = "Moderate Risk"
                                elif suspicion < 0.8:
                                    risk_emoji = "🟠"
                                    risk_text = "High Risk"
                                else:
                                    risk_emoji = "🔴"
                                    risk_text = "Very High Risk"
                                
                                # Display the ELA heatmap
                                st.image(
                                    row["ela_png"],
                                    caption=f"{filename}\n{risk_emoji} Suspicion: {suspicion:.3f} ({risk_text})",
                                    use_container_width=True,
                                )
                                
                                # Add individual metrics below each heatmap
                                with st.container():
                                    metric_cols = st.columns(3)
                                    with metric_cols[0]:
                                        st.caption(f"ELA: {row['ela_score']:.3f}")
                                    with metric_cols[1]:
                                        st.caption(f"FFT: {row['fft_score']:.3f}")
                                    with metric_cols[2]:
                                        face_icon = "👤" if row["face_conf"] > 0.5 else "⚪"
                                        st.caption(f"Face: {face_icon}")
                
                # Add heatmap interpretation guide
                with st.expander("🎨 How to read ELA heatmaps", expanded=False):
                    st.markdown("""
                    **Error Level Analysis (ELA) Heatmap Interpretation:**
                    
                    🔴 **Bright/Hot areas**: Indicate high compression differences
                    - May suggest recent edits or modifications
                    - Different compression levels from original
                    - Potential copy-paste or splicing
                    
                    🔵 **Dark/Cool areas**: Show consistent compression
                    - Likely original or uniformly compressed
                    - No recent modifications detected
                    
                    **What to look for:**
                    - **Edges with different brightness**: Possible object insertion
                    - **Rectangular patterns**: May indicate copied regions
                    - **Inconsistent textures**: Different source images combined
                    - **Bright outlines around objects**: Recent additions or alterations
                    
                    **Note**: Some legitimate operations (resizing, format conversion) can also 
                    create ELA patterns, so consider the context and other scores.
                    """)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

# Add footer with tech stack
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown(
        """
        <div style='text-align: center; padding: 20px;'>
            <h4>🛰️ Deepfake Intelligence Radar</h4>
            <p style='color: gray;'>Built with <strong>Daft DataFrame Engine</strong> + <strong>Daytona Dev Platform</strong></p>
            <p style='color: gray; font-size: 12px;'>Streamlit • OpenCV • MediaPipe • NumPy</p>
        </div>
        """,
        unsafe_allow_html=True
    )

