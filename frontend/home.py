import streamlit as st

def show_home():
    # --- Custom CSS for the Professional Look ---
    st.markdown("""
    <style>
        /* Global font adjustments */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 600;
        }
        
        /* Hero Section Styling */
        .hero-container {
            background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
            border-left: 5px solid #00C853; /* Green Accent */
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 40px;
            box-shadow: 0 4px 12px rgba(0,200,83, 0.15);
        }
        .hero-title {
            font-size: 3rem;
            color: #ffffff;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            color: #b0b0b0;
            font-weight: 300;
        }
        
        /* Feature Cards */
        .feature-card {
            background-color: #1E1E1E;
            padding: 25px;
            border-radius: 8px;
            border: 1px solid #333;
            height: 100%;
            transition: transform 0.2s;
        }
        .feature-card:hover {
            border-color: #00C853; /* Hover Green */
            transform: translateY(-2px);
        }
        .feature-title {
            color: #00C853;
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .feature-text {
            color: #e0e0e0;
            font-size: 0.95rem;
            line-height: 1.5;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Hero Section ---
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">HIRE Ready AI</div>
        <div class="hero-subtitle">
            Advanced Resume Architecture & Contextual Tailoring System
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Product Description (No Emojis) ---
    st.markdown("### Core Capabilities")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Contextual Analysis</div>
            <div class="feature-text">
                Ingests candidate documentation (PDF) and Job Descriptions (JD) to perform semantic gap analysis using Large Language Models.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Intelligent Retrieval</div>
            <div class="feature-text">
                Leverages RAG (Retrieval-Augmented Generation) to query your GitHub history and extract projects that mathematically align with the target role.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Adaptive Synthesis</div>
            <div class="feature-text">
                Reconstructs your professional experience section, dynamically rewriting bullet points to match the syntactic style and keywords of the employer.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacer
    st.markdown("---")
    st.markdown("*Select 'Tailor Resume' from the navigation menu to begin.*")

# Execute
show_home()