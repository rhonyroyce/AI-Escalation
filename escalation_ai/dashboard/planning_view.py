"""
Planning & Actions view for the Escalation AI dashboard.

Extracted from streamlit_app.py. Provides render_planning_actions() which
renders a 3-tab page: What-If Simulator, Action Tracker, and Learning-Based Actions.
"""

import streamlit as st
import pandas as pd

from escalation_ai.dashboard.shared_helpers import render_spectacular_header


def _render_whatif_tab(df):
    """Render the What-If Simulator tab with scenario sliders and projections."""
    st.markdown("#### Adjust parameters to simulate impact on escalation metrics")

    recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    friction_sum = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 3000
    cost_sum = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else 375000

    # Calculate current lesson coverage
    lesson_col = None
    for col in ['tickets_data_lessons_learned_title', 'tickets_data_lessons_learned_preventive_actions']:
        if col in df.columns:
            lesson_col = col
            break
    current_lesson_coverage = (df[lesson_col].notna().sum() / len(df) * 100) if lesson_col else 0

    # Calculate similarity-based recurrence
    similarity_recurrence = (df['Similar_Ticket_Count'] > 0).mean() * 100 if 'Similar_Ticket_Count' in df.columns else recurrence_rate * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🎛️ Scenario Parameters")
        staffing = st.slider("Staffing Changes (engineers)", -5, 10, 0, key="whatif_staff")
        training = st.slider("Training Impact (% error reduction)", 0, 50, 0, key="whatif_train")
        volume = st.slider("Volume Changes (%)", -30, 50, 0, key="whatif_vol")
        process = st.slider("Process Improvements (% efficiency)", 0, 40, 0, key="whatif_proc")

        st.markdown("##### 📚 Learning & Knowledge Parameters")
        lesson_improvement = st.slider("Lesson Documentation Rate (+%)", 0, 50, 0, key="whatif_lesson",
                                      help=f"Current: {current_lesson_coverage:.0f}% coverage")
        lesson_application = st.slider("Lesson Application Effectiveness (%)", 0, 80, 0, key="whatif_apply",
                                      help="How effectively lessons prevent recurrence")

    with col2:
        _render_whatif_projections(
            avg_resolution, recurrence_rate, similarity_recurrence,
            friction_sum, cost_sum, len(df),
            staffing, training, volume, process,
            lesson_improvement, lesson_application
        )


def _render_whatif_projections(avg_resolution, recurrence_rate, similarity_recurrence,
                                friction_sum, cost_sum, n_records,
                                staffing, training, volume, process,
                                lesson_improvement, lesson_application):
    """Render projected impact metrics based on scenario parameters."""
    st.markdown("##### 📊 Projected Impact")

    staff_factor = 1 - (staffing * 0.03)
    training_factor = 1 - (training / 100)
    volume_factor = 1 + (volume / 100)
    process_factor = 1 - (process / 100)

    lesson_coverage_factor = 1 - (lesson_improvement / 100 * 0.3)
    lesson_application_factor = 1 - (lesson_application / 100 * 0.5)

    proj_resolution = avg_resolution * staff_factor * process_factor
    proj_recurrence = recurrence_rate * training_factor * lesson_coverage_factor * lesson_application_factor
    proj_similarity_recurrence = similarity_recurrence * lesson_application_factor * training_factor
    proj_friction = friction_sum * volume_factor * process_factor / n_records
    proj_cost = cost_sum * volume_factor * staff_factor * process_factor * lesson_application_factor / n_records

    metrics = [
        ("Resolution Time", f"{avg_resolution:.1f}d", f"{proj_resolution:.1f}d", proj_resolution < avg_resolution),
        ("Recurrence Rate", f"{recurrence_rate*100:.1f}%", f"{proj_recurrence*100:.1f}%", proj_recurrence < recurrence_rate),
        ("Similar Issue Rate", f"{similarity_recurrence:.1f}%", f"{proj_similarity_recurrence:.1f}%", proj_similarity_recurrence < similarity_recurrence),
        ("Avg Friction", f"{friction_sum/n_records:.1f}", f"{proj_friction:.1f}", proj_friction < friction_sum/n_records),
        ("Avg Cost", f"${cost_sum/n_records:,.0f}", f"${proj_cost:,.0f}", proj_cost < cost_sum/n_records)
    ]

    for name, baseline, projected, is_better in metrics:
        color = '#22c55e' if is_better else '#ef4444'
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 12px; margin: 8px 0;
                    display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8;">{name}</span>
            <div>
                <span style="color: #64748b; text-decoration: line-through; margin-right: 10px;">{baseline}</span>
                <span style="color: {color}; font-weight: 700;">{projected}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Show potential savings
    if lesson_improvement > 0 or lesson_application > 0:
        base_annual_cost = cost_sum * 4
        projected_annual_cost = proj_cost * n_records * 4
        savings = base_annual_cost - projected_annual_cost
        if savings > 0:
            st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); border-radius: 8px; padding: 15px; margin-top: 15px; border: 1px solid rgba(34, 197, 94, 0.3);">
                <div style="color: #86efac; font-size: 0.85rem;">💰 Projected Annual Savings from Learning Improvements</div>
                <div style="color: #22c55e; font-size: 1.8rem; font-weight: 700;">${savings:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_action_tracker_tab(df):
    """Render the Action Tracker tab with systemic issues and initiative CRUD."""
    from escalation_ai.dashboard.streamlit_app import generate_strategic_recommendations

    # Top Systemic Issues
    st.markdown("#### 🎯 Top 5 Systemic Issues")
    st.markdown("*Data-driven analysis of highest-impact recurring problems*")

    if 'AI_Category' in df.columns:
        issue_analysis = df.groupby('AI_Category').agg({
            'Financial_Impact': ['sum', 'count'],
            'AI_Recurrence_Risk': 'mean' if 'AI_Recurrence_Risk' in df.columns else 'count'
        }).reset_index()
        issue_analysis.columns = ['Category', 'Total_Cost', 'Count', 'Recurrence']
        issue_analysis = issue_analysis.sort_values('Total_Cost', ascending=False).head(5)

        for idx, row in issue_analysis.iterrows():
            rec = generate_strategic_recommendations(df[df['AI_Category'] == row['Category']])
            fix = rec[0]['description'] if rec else "Implement process improvements"

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 16px; margin: 10px 0;
                        border-left: 4px solid #ef4444;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #e2e8f0; font-weight: 700; font-size: 1.1rem;">{row['Category']}</span>
                        <span style="color: #64748b; margin-left: 15px;">{row['Count']} tickets</span>
                    </div>
                    <span style="color: #ef4444; font-weight: 700; font-size: 1.2rem;">${row['Total_Cost']:,.0f}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">
                    <b>Recommended Fix:</b> {fix[:150]}...
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Initiative Management
    st.markdown("---")
    st.markdown("#### 📋 Initiative Management")

    if 'action_items' not in st.session_state:
        st.session_state.action_items = []

    # Add new initiative
    with st.expander("➕ Add New Initiative"):
        new_title = st.text_input("Initiative Title", key="new_init_title")
        new_priority = st.selectbox("Priority", ["P1", "P2", "P3"], key="new_init_priority")
        new_owner = st.text_input("Owner", key="new_init_owner")
        new_due = st.date_input("Due Date", key="new_init_due")

        if st.button("Add Initiative", key="add_init_btn"):
            if new_title:
                st.session_state.action_items.append({
                    'title': new_title,
                    'priority': new_priority,
                    'owner': new_owner,
                    'due': str(new_due),
                    'status': 'Not Started',
                    'progress': 0
                })
                st.success(f"Added: {new_title}")
                st.rerun()

    # Display initiatives
    for i, item in enumerate(st.session_state.action_items):
        priority_color = '#ef4444' if item['priority'] == 'P1' else '#f97316' if item['priority'] == 'P2' else '#3b82f6'
        status_color = '#22c55e' if item['status'] == 'Complete' else '#f97316' if item['status'] == 'In Progress' else '#64748b'

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 12px;
                        border-left: 4px solid {priority_color};">
                <span style="background: {priority_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">{item['priority']}</span>
                <span style="color: #e2e8f0; font-weight: 600; margin-left: 10px;">{item['title']}</span>
                <div style="color: #64748b; font-size: 0.8rem; margin-top: 5px;">
                    Owner: {item['owner']} | Due: {item['due']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            new_status = st.selectbox("Status", ["Not Started", "In Progress", "Complete"],
                                     index=["Not Started", "In Progress", "Complete"].index(item['status']),
                                     key=f"status_{i}", label_visibility="collapsed")
            if new_status != item['status']:
                st.session_state.action_items[i]['status'] = new_status
                st.rerun()
        with col3:
            if st.button("🗑️", key=f"del_{i}"):
                st.session_state.action_items.pop(i)
                st.rerun()


def _render_learning_actions_tab(df):
    """Render Learning-Based Actions tab with AI recommendations."""
    st.markdown("#### 📚 AI-Generated Actions from Learning Analysis")
    st.markdown("*Prioritized actions based on similarity patterns and lesson effectiveness*")

    # Find lesson column
    lesson_col = None
    for col in ['tickets_data_lessons_learned_title', 'tickets_data_lessons_learned_preventive_actions', 'Lesson_Title']:
        if col in df.columns:
            lesson_col = col
            break

    if not (lesson_col and 'Similar_Ticket_Count' in df.columns and 'AI_Category' in df.columns):
        st.info("📊 This analysis requires:\n- Lessons learned data column\n- Similar_Ticket_Count from similarity analysis\n- AI_Category classification\n\nRun the full analysis pipeline to enable learning-based action recommendations.")
        return

    action_items = _calculate_learning_actions(df, lesson_col)

    if action_items:
        _render_learning_actions_summary(action_items)
        _render_learning_actions_list(action_items)
    else:
        st.success("✅ No critical learning-based actions identified. Categories are performing well!")


def _calculate_learning_actions(df, lesson_col):
    """Calculate learning-based action items from data analysis."""
    df_analysis = df.copy()
    df_analysis['Has_Lesson'] = df_analysis[lesson_col].notna() & (df_analysis[lesson_col].astype(str).str.strip() != '')
    df_analysis['Has_Similar'] = df_analysis['Similar_Ticket_Count'] > 0

    action_items = []

    for cat in df_analysis['AI_Category'].dropna().unique():
        cat_df = df_analysis[df_analysis['AI_Category'] == cat]
        if len(cat_df) < 3:
            continue

        total = len(cat_df)
        with_lessons = cat_df['Has_Lesson'].sum()
        recurring_with_lessons = ((cat_df['Has_Similar']) & (cat_df['Has_Lesson'])).sum()

        lesson_coverage = (with_lessons / total * 100) if total > 0 else 0
        recurrence_rate = (cat_df['Has_Similar'].sum() / total * 100) if total > 0 else 0
        cat_cost = cat_df['Financial_Impact'].sum() if 'Financial_Impact' in cat_df.columns else 0

        priority = None
        action_type = None
        action_desc = None
        potential_savings = 0

        # Case 1: High recurrence, low lesson coverage
        if recurrence_rate > 50 and lesson_coverage < 30:
            priority = "P1"
            action_type = "📝 Document Lessons"
            action_desc = f"Only {lesson_coverage:.0f}% of tickets have lessons but {recurrence_rate:.0f}% are recurring. Mandate lesson documentation for all resolved tickets."
            potential_savings = cat_cost * 0.3

        # Case 2: Lessons exist but not working
        elif with_lessons > 0 and recurring_with_lessons > with_lessons * 0.4:
            priority = "P1"
            action_type = "🔄 Improve Lesson Application"
            action_desc = f"{recurring_with_lessons} issues recurred despite having lessons. Review lesson quality and ensure teams are applying documented solutions."
            potential_savings = cat_cost * 0.25

        # Case 3: Moderate recurrence, some lessons
        elif recurrence_rate > 30 and lesson_coverage > 30 and lesson_coverage < 70:
            priority = "P2"
            action_type = "📢 Knowledge Sharing"
            action_desc = f"Lessons exist ({lesson_coverage:.0f}% coverage) but recurrence is {recurrence_rate:.0f}%. Improve cross-team knowledge sharing and training."
            potential_savings = cat_cost * 0.15

        # Case 4: Low lesson coverage, moderate issues
        elif lesson_coverage < 40 and total >= 10:
            priority = "P2"
            action_type = "📝 Document Lessons"
            action_desc = f"Low lesson coverage ({lesson_coverage:.0f}%) for {total} tickets. Establish lesson documentation as part of resolution workflow."
            potential_savings = cat_cost * 0.2

        # Case 5: Good lesson coverage but could improve consistency
        elif 'Resolution_Consistency' in df.columns:
            inconsistent = cat_df['Resolution_Consistency'].str.contains('Inconsistent', na=False).sum()
            if inconsistent > total * 0.2:
                priority = "P3"
                action_type = "⚙️ Standardize Resolution"
                action_desc = f"{inconsistent} tickets ({inconsistent/total*100:.0f}%) have inconsistent resolutions. Create standard operating procedures."
                potential_savings = cat_cost * 0.1

        if priority:
            action_items.append({
                'category': cat,
                'priority': priority,
                'action_type': action_type,
                'description': action_desc,
                'ticket_count': total,
                'recurrence_rate': recurrence_rate,
                'lesson_coverage': lesson_coverage,
                'potential_savings': potential_savings,
                'cost': cat_cost
            })

    # Sort by priority then by potential savings
    priority_order = {'P1': 0, 'P2': 1, 'P3': 2}
    action_items.sort(key=lambda x: (priority_order.get(x['priority'], 3), -x['potential_savings']))

    return action_items


def _render_learning_actions_summary(action_items):
    """Render summary metrics for learning actions."""
    col1, col2, col3 = st.columns(3)
    with col1:
        p1_count = sum(1 for a in action_items if a['priority'] == 'P1')
        st.metric("🔴 P1 Actions", p1_count, help="Critical - immediate action needed")
    with col2:
        p2_count = sum(1 for a in action_items if a['priority'] == 'P2')
        st.metric("🟠 P2 Actions", p2_count, help="Important - plan within 30 days")
    with col3:
        total_savings = sum(a['potential_savings'] for a in action_items)
        st.metric("💰 Potential Savings", f"${total_savings:,.0f}")

    st.markdown("---")


def _render_learning_actions_list(action_items):
    """Render the list of learning-based action items."""
    for item in action_items[:10]:
        priority_color = '#ef4444' if item['priority'] == 'P1' else '#f97316' if item['priority'] == 'P2' else '#3b82f6'

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 16px; margin: 12px 0;
                    border-left: 4px solid {priority_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                <div>
                    <span style="background: {priority_color}; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">{item['priority']}</span>
                    <span style="color: #60a5fa; margin-left: 10px; font-size: 0.9rem;">{item['action_type']}</span>
                </div>
                <span style="color: #22c55e; font-weight: 600;">Save ${item['potential_savings']:,.0f}</span>
            </div>
            <div style="color: #e2e8f0; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px;">{item['category']}</div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px;">{item['description']}</div>
            <div style="display: flex; gap: 20px; font-size: 0.8rem; color: #64748b;">
                <span>📊 {item['ticket_count']} tickets</span>
                <span>🔄 {item['recurrence_rate']:.0f}% recurrence</span>
                <span>📝 {item['lesson_coverage']:.0f}% lesson coverage</span>
                <span>💰 ${item['cost']:,.0f} total cost</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Export actions as table
    with st.expander("📋 Export Action Items"):
        actions_df = pd.DataFrame(action_items)
        actions_df = actions_df.rename(columns={
            'category': 'Category',
            'priority': 'Priority',
            'action_type': 'Action Type',
            'description': 'Description',
            'ticket_count': 'Tickets',
            'recurrence_rate': 'Recurrence %',
            'lesson_coverage': 'Lesson Coverage %',
            'potential_savings': 'Potential Savings',
            'cost': 'Total Cost'
        })
        st.dataframe(actions_df, use_container_width=True, hide_index=True)


def render_planning_actions(df):
    """Render the Planning and Actions page with 3 tabs.

    Tabs:
    1. What-If Simulator: Extended version with lesson-specific parameters
       (lesson coverage, similarity utilization) in addition to standard
       staffing/training/volume/process sliders. Projects impact on
       recurrence, resolution, friction, and cost.
    2. Action Tracker: Delegates to render_action_tracker() for systemic
       issue tracking and initiative CRUD.
    3. Learning-Based Actions: AI recommendations from
       generate_ai_lesson_recommendations(), learning grades table, and
       at-risk category identification for targeted interventions.

    Args:
        df: Processed DataFrame with standard columns plus lesson columns.
    """
    render_spectacular_header("Planning & Actions", "Scenario modeling and initiative tracking", "🎯")

    tabs = st.tabs(["🔮 What-If Simulator", "📋 Action Tracker", "📚 Learning-Based Actions"])

    with tabs[0]:
        _render_whatif_tab(df)

    with tabs[1]:
        _render_action_tracker_tab(df)

    with tabs[2]:
        _render_learning_actions_tab(df)
