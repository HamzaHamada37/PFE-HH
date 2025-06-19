from django.urls import path
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('overview/', views.overview, name='overview'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_file, name='upload_file'),
    path('process/<int:file_id>/', views.process_file, name='process_file'),
    path('analysis/<int:analysis_id>/', views.view_analysis, name='view_analysis'),
    path('debug-analysis/<int:analysis_id>/', views.debug_analysis, name='debug_analysis'),
    path('download-data/<int:analysis_id>/', views.download_cleaned_data, name='download_cleaned_data'),
    path('regenerate/<int:file_id>/', views.regenerate_analysis, name='regenerate_analysis'),
    path('delete/<int:file_id>/', views.delete_file, name='delete_file'),
    path('ai-agent/', views.ai_agent, name='ai_agent'),
    path('client-overview/', views.client_overview, name='client_overview'),
    path('client-overview/<str:client_name>/', views.client_detail, name='client_detail'),
    path('team-overview/', views.team_overview, name='team_overview'),
    path('test-logo/', TemplateView.as_view(template_name='test_logo.html'), name='test_logo'),

    # Export URLs for Feature 1
    path('export/csv/', views.export_csv, name='export_csv'),
    path('export/excel/', views.export_excel, name='export_excel'),
    path('export/pdf/', views.export_pdf, name='export_pdf'),

    # Survey Export URLs
    path('export/survey-csv/', views.export_survey_csv, name='export_survey_csv'),
    path('export/survey-excel/', views.export_survey_excel, name='export_survey_excel'),
    path('export/team-overview-pdf/', views.export_team_overview_pdf, name='export_team_overview_pdf'),

    # Notes URLs for Feature 2
    path('client-note/<str:client_name>/', views.get_client_note, name='get_client_note'),
    path('client-note/<str:client_name>/save/', views.save_client_note, name='save_client_note'),

    # Chatbot API URLs
    path('api/chat/message/', views.chat_message, name='chat_message'),
    path('api/chat/sessions/', views.chat_sessions, name='chat_sessions'),
    path('api/chat/history/<str:session_id>/', views.chat_history, name='chat_history'),
    path('api/chat/delete/<str:session_id>/', views.delete_chat_session, name='delete_chat_session'),
    path('api/chat/data-summary/', views.chat_data_summary, name='chat_data_summary'),

    # RAG API URLs
    path('api/rag/refresh/', views.refresh_rag_index, name='refresh_rag_index'),
    path('api/rag/stats/', views.rag_stats, name='rag_stats'),
    path('api/rag/search/', views.search_knowledge_base, name='search_knowledge_base'),
    path('api/rag/comprehensive-index/', views.comprehensive_data_index, name='comprehensive_data_index'),
    path('api/rag/debug/', views.debug_rag_retrieval, name='debug_rag_retrieval'),
]
