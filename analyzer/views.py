from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import JiraFile, AnalysisResult, ClientNote, ChatSession, ChatMessage, SurveyFile, SurveyAnalysis
from .forms import JiraFileUploadForm, SurveyFileUploadForm
from .utils import process_jira_file, process_survey_file, validate_survey_file_structure  # This will be implemented later with NLP
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import pandas as pd
from collections import defaultdict
import json
from django.utils import timezone
from datetime import timedelta
import urllib.parse
import os
from django.conf import settings
import csv
from io import BytesIO
from .chatbot_service import ChatbotService
# PDF imports will be loaded dynamically in the export_pdf view

@login_required
def ai_agent(request):
    """Main AI Agent page with chat interface"""
    return render(request, 'analyzer/ai_agent.html')

@login_required
def dashboard(request):
    # Order files by analysis_date (newest first), fallback to uploaded_at if analysis_date is null
    jira_files = JiraFile.objects.filter(user=request.user).order_by('-analysis_date', '-uploaded_at')

    # Get client metrics summary data
    client_count = 0
    total_tickets = 0
    client_metrics_summary = None
    actionable_insights = None

    # Get all analysis results for the user, ordered by analysis_date (newest first), fallback to creation date
    analysis_results = AnalysisResult.objects.filter(
        jira_file__user=request.user
    ).order_by('-jira_file__analysis_date', '-created_at')

    # Handle comparison if requested
    comparison_data = None
    current_analysis = None
    comparison_analysis = None
    available_analyses = None
    has_comparisons = False

    if analysis_results.count() >= 2:
        has_comparisons = True
        available_analyses = analysis_results

        if request.GET.get('compare_current') and request.GET.get('compare_with'):
            try:
                current_id = int(request.GET.get('compare_current'))
                compare_id = int(request.GET.get('compare_with'))
                current_analysis = get_object_or_404(
                    AnalysisResult,
                    id=current_id,
                    jira_file__user=request.user
                )
                comparison_analysis = get_object_or_404(
                    AnalysisResult,
                    id=compare_id,
                    jira_file__user=request.user
                )
                comparison_data = calculate_comparison_metrics(current_analysis, comparison_analysis)
            except (ValueError, AnalysisResult.DoesNotExist):
                messages.error(request, "Invalid comparison analyses selected.")

    # Get the latest analysis for actionable insights and client metrics
    latest_analysis = analysis_results.first()
    if latest_analysis:
        if hasattr(latest_analysis, 'actionable_insights') and latest_analysis.actionable_insights:
            actionable_insights = latest_analysis.actionable_insights

        # Debug output for client metrics
        if hasattr(latest_analysis, 'client_metrics') and latest_analysis.client_metrics:
            print(f"Latest analysis client metrics: {latest_analysis.client_metrics}")
            print(f"Number of clients in latest analysis: {len(latest_analysis.client_metrics)}")

    # Initialize data structures to store client metrics
    all_clients = set()
    latest_metrics = {}
    global_sentiment_data = {}

    # Process each analysis result to extract client metrics and build global sentiment tracking
    for analysis in analysis_results:
        if not analysis.client_metrics:
            continue

        # Use analysis_date if available, otherwise fall back to created_at
        if analysis.jira_file.analysis_date:
            analysis_date = analysis.jira_file.analysis_date
        else:
            analysis_date = analysis.created_at.date()

        # Process each client's metrics
        for client_name, metrics in analysis.client_metrics.items():
            all_clients.add(client_name)

            # Only update latest_metrics if we haven't seen this client before (to get latest metrics)
            if client_name not in latest_metrics:
                latest_metrics[client_name] = metrics

            # Build global sentiment tracking data
            if client_name not in global_sentiment_data:
                global_sentiment_data[client_name] = []

            global_sentiment_data[client_name].append({
                'date': analysis_date,
                'sentiment': metrics.get('sentiment', 0),
                'tickets': metrics.get('Tickets', 0),
                'resolution_time': metrics.get('Avg_Resolution_Time_Days', 0),
                'analysis_id': analysis.id
            })

    # Calculate sentiment trends and current status for each client
    sentiment_analysis_data = {}
    for client_name in all_clients:
        if client_name in global_sentiment_data and global_sentiment_data[client_name]:
            # Sort by date (newest first)
            client_data = sorted(global_sentiment_data[client_name], key=lambda x: x['date'], reverse=True)

            current_sentiment = client_data[0]['sentiment']
            last_update = client_data[0]['date']

            # Calculate trend if we have multiple data points
            trend = 'stable'
            if len(client_data) > 1:
                # Compare current sentiment with previous sentiment
                previous_sentiment = client_data[1]['sentiment']
                sentiment_change = current_sentiment - previous_sentiment

                if sentiment_change > 0.1:
                    trend = 'improving'
                elif sentiment_change < -0.1:
                    trend = 'declining'
                else:
                    trend = 'stable'

            # Determine severity level
            severity = 'neutral'
            if current_sentiment < -0.5:
                severity = 'critical'
            elif current_sentiment < -0.4:
                severity = 'high'
            elif current_sentiment < -0.2:
                severity = 'medium'
            elif current_sentiment < 0:
                severity = 'low'
            else:
                severity = 'neutral'

            sentiment_analysis_data[client_name] = {
                'current_sentiment': current_sentiment,
                'trend': trend,
                'severity': severity,
                'last_update': last_update.strftime('%Y-%m-%d'),  # Convert date to string
                'tickets': client_data[0]['tickets'],
                'resolution_time': client_data[0]['resolution_time'],
                'data_points': len(client_data)
            }

    # Feature 4: Filter sentiment analysis data for last 5 updated clients with negative sentiment
    # Sort clients by last update date (newest first) and filter for negative sentiment
    negative_sentiment_clients = []
    for client_name, data in sentiment_analysis_data.items():
        if data['current_sentiment'] < 0:  # Only negative sentiment
            negative_sentiment_clients.append((client_name, data['last_update'], data))

    # Sort by last update date (newest first) and take top 5
    negative_sentiment_clients.sort(key=lambda x: x[1], reverse=True)
    top_5_negative_clients = negative_sentiment_clients[:5]

    # Create filtered sentiment data for the dashboard table
    filtered_sentiment_data = {}
    for client_name, last_update, data in top_5_negative_clients:
        filtered_sentiment_data[client_name] = data

    # Count total clients
    client_count = len(all_clients)

    # Calculate total tickets by summing issue_count from all processed files
    # This ensures we count all tickets across all files
    processed_analysis_results = analysis_results.filter(jira_file__processed=True)
    total_tickets = sum(analysis.issue_count for analysis in processed_analysis_results)

    # Prepare client metrics summary for charts
    client_metrics_summary = {
        'sentiment': {
            'positive': 0,
            'neutral': 0,
            'negative': 0
        },
        'resolution_time': [],
        'client_impact': []
    }

    # Process client metrics for summary
    for client, metrics in latest_metrics.items():
        # Sentiment counts
        if metrics.get('sentiment', 0) > 0.3:
            client_metrics_summary['sentiment']['positive'] += 1
        elif metrics.get('sentiment', 0) < -0.3:
            client_metrics_summary['sentiment']['negative'] += 1
        else:
            client_metrics_summary['sentiment']['neutral'] += 1

        # Resolution time data
        if 'Avg_Resolution_Time_Days' in metrics:
            client_metrics_summary['resolution_time'].append({
                'client': client,
                'days': metrics['Avg_Resolution_Time_Days']
            })

        # Customer Experience Score data
        if 'Client_Impact' in metrics:
            client_metrics_summary['client_impact'].append({
                'client': client,
                'impact': float(metrics['Client_Impact'])
            })
        # Fallback if Client_Impact is missing but we have Customer_Experience_Score
        elif 'Customer_Experience_Score' in metrics:
            client_metrics_summary['client_impact'].append({
                'client': client,
                'impact': float(metrics['Customer_Experience_Score'])
            })
        # Fallback if both are missing but we have sentiment
        elif 'sentiment' in metrics:
            client_metrics_summary['client_impact'].append({
                'client': client,
                'impact': float(metrics.get('sentiment', 0))
            })

    # Ensure client metrics summary is properly formatted
    if client_metrics_summary and 'client_impact' in client_metrics_summary:
        # Make sure we have at least one client with impact data
        if len(client_metrics_summary['client_impact']) == 0:
            # If no clients have impact data, add a placeholder
            client_metrics_summary['client_impact'].append({
                'client': 'No Data',
                'impact': 0
            })



    context = {
        'jira_files': jira_files,
        'client_count': client_count,
        'total_tickets': total_tickets,
        'client_metrics_summary': client_metrics_summary,
        'latest_analysis': latest_analysis,
        'actionable_insights': actionable_insights,
        'available_analyses': available_analyses,
        'comparison_data': comparison_data,
        'current_analysis': current_analysis,
        'comparison_analysis': comparison_analysis,
        'has_comparisons': has_comparisons,
        'sentiment_analysis_data': filtered_sentiment_data,  # Feature 4: Use filtered data
    }

    return render(request, 'analyzer/dashboard.html', context)

@login_required
def upload_file(request):
    if request.method == 'POST':
        form = JiraFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            jira_file = form.save(commit=False)
            jira_file.user = request.user
            # The analysis_date is already handled by the form since it's in the Meta fields
            jira_file.save()

            messages.success(request, f"File uploaded successfully with analysis date {jira_file.analysis_date}. Processing will begin shortly.")
            return redirect('process_file', file_id=jira_file.id)
    else:
        form = JiraFileUploadForm()

    return render(request, 'analyzer/upload.html', {'form': form})

@login_required
def process_file(request, file_id):
    jira_file = get_object_or_404(JiraFile, id=file_id, user=request.user)

    try:
        # Process the file using the NLP utility
        analysis_results, client_metrics_df = process_jira_file(jira_file)

        # Delete any existing analysis results for this file
        jira_file.analysis_results.all().delete()

        # Create a new analysis result with the processed data
        analysis = AnalysisResult.objects.create(
            jira_file=jira_file,
            **analysis_results
        )

        # Mark the file as processed
        jira_file.processed = True
        jira_file.save()

        # Debug output
        print("Analysis results saved to database:")
        print(f"Client metrics: {analysis.client_metrics}")
        print(f"Client metrics type: {type(analysis.client_metrics)}")
        print(f"Number of clients: {len(analysis.client_metrics) if analysis.client_metrics else 0}")

        messages.success(request, "File processed successfully!")
        return redirect('view_analysis', analysis_id=analysis.id)

    except Exception as e:
        import traceback
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        messages.error(request, f"Error processing file: {str(e)}")
        return redirect('dashboard')

def calculate_comparison_metrics(current_analysis, comparison_analysis):
    """Calculate comparison metrics between two analyses - only for common clients"""
    comparison_data = {
        'client_comparisons': {},
        'overall_changes': {},
        'has_common_clients': False
    }

    current_metrics = current_analysis.client_metrics or {}
    comparison_metrics = comparison_analysis.client_metrics or {}

    # Get only common clients from both analyses (intersection)
    common_clients = set(current_metrics.keys()) & set(comparison_metrics.keys())

    if not common_clients:
        return comparison_data

    comparison_data['has_common_clients'] = True

    for client in common_clients:
        current_client = current_metrics[client]
        comparison_client = comparison_metrics[client]

        # Calculate changes for each metric
        client_comparison = {}

        # Sentiment change
        current_sentiment = current_client.get('sentiment', 0)
        comparison_sentiment = comparison_client.get('sentiment', 0)
        sentiment_change = current_sentiment - comparison_sentiment
        client_comparison['sentiment_change'] = sentiment_change

        # Resolution time change
        current_resolution = current_client.get('Avg_Resolution_Time_Days', 0)
        comparison_resolution = comparison_client.get('Avg_Resolution_Time_Days', 0)
        resolution_change = current_resolution - comparison_resolution
        client_comparison['resolution_time_change'] = resolution_change

        # Customer experience score change
        current_impact = current_client.get('Client_Impact', 0)
        comparison_impact = comparison_client.get('Client_Impact', 0)
        impact_change = current_impact - comparison_impact
        client_comparison['client_impact_change'] = impact_change

        # Ticket count change
        current_tickets = current_client.get('Tickets', 0)
        comparison_tickets = comparison_client.get('Tickets', 0)
        tickets_change = current_tickets - comparison_tickets
        client_comparison['tickets_change'] = tickets_change

        # Store current and comparison values for display
        client_comparison['current'] = current_client
        client_comparison['comparison'] = comparison_client

        comparison_data['client_comparisons'][client] = client_comparison

    # Calculate overall changes (only for common clients)
    if common_clients:
        # Average sentiment change
        sentiment_changes = [comp['sentiment_change'] for comp in comparison_data['client_comparisons'].values()]
        comparison_data['overall_changes']['avg_sentiment_change'] = sum(sentiment_changes) / len(sentiment_changes) if sentiment_changes else 0

        # Average resolution time change
        resolution_changes = [comp['resolution_time_change'] for comp in comparison_data['client_comparisons'].values()]
        comparison_data['overall_changes']['avg_resolution_change'] = sum(resolution_changes) / len(resolution_changes) if resolution_changes else 0

        # Average client impact change
        impact_changes = [comp['client_impact_change'] for comp in comparison_data['client_comparisons'].values()]
        comparison_data['overall_changes']['avg_impact_change'] = sum(impact_changes) / len(impact_changes) if impact_changes else 0

    return comparison_data

@login_required
def view_analysis(request, analysis_id):
    analysis = get_object_or_404(AnalysisResult, id=analysis_id)
    jira_file = analysis.jira_file

    # Ensure the user has permission to view this analysis
    if jira_file.user != request.user:
        messages.error(request, "You don't have permission to view this analysis.")
        return redirect('dashboard')



    # Debug: Print client metrics to console
    print("Client Metrics in view_analysis:", analysis.client_metrics)
    print("Client Metrics type:", type(analysis.client_metrics))

    context = {
        'analysis': analysis,
        'jira_file': jira_file,
    }
    return render(request, 'analyzer/view_analysis.html', context)

def home(request):
    if request.user.is_authenticated:
        return redirect('overview')
    return render(request, 'analyzer/home.html')

@login_required
def overview(request):
    """
    Display the overview page with information about NLP analysis and client metrics.
    This serves as the landing page after authentication.
    """
    return render(request, 'analyzer/overview.html')

@login_required
def regenerate_analysis(request, file_id):
    """Regenerate analysis for an existing file"""
    jira_file = get_object_or_404(JiraFile, id=file_id, user=request.user)

    try:
        # Process the file using the NLP utility
        analysis_results, client_metrics_df = process_jira_file(jira_file)

        # Delete existing analysis results
        jira_file.analysis_results.all().delete()

        # Create new analysis result
        analysis = AnalysisResult.objects.create(
            jira_file=jira_file,
            **analysis_results
        )

        # Debug output
        print("Analysis regenerated with client metrics:")
        print(f"Client metrics: {analysis.client_metrics}")
        print(f"Client metrics type: {type(analysis.client_metrics)}")

        messages.success(request, "Analysis regenerated successfully!")
        return redirect('view_analysis', analysis_id=analysis.id)

    except Exception as e:
        messages.error(request, f"Error regenerating analysis: {str(e)}")
        return redirect('dashboard')

@login_required
def debug_analysis(request, analysis_id):
    """Debug view to directly output analysis data as JSON"""
    analysis = get_object_or_404(AnalysisResult, id=analysis_id)

    # Ensure the user has permission to view this analysis
    if analysis.jira_file.user != request.user:
        return JsonResponse({"error": "Permission denied"}, status=403)

    # Return all analysis data as JSON
    return JsonResponse({
        "analysis_id": analysis.id,
        "file_name": analysis.jira_file.file.name,
        "issue_count": analysis.issue_count,
        "ticket_types": analysis.ticket_types,
        "priority_distribution": analysis.priority_distribution,
        "status_distribution": analysis.status_distribution,
        "common_themes": analysis.common_themes,
        "sentiment_analysis": analysis.sentiment_analysis,
        "client_metrics": analysis.client_metrics,
        "client_metrics_type": str(type(analysis.client_metrics))
    })

@login_required
def download_cleaned_data(request, analysis_id):
    """Download the cleaned JIRA data with impact scores as CSV"""
    analysis = get_object_or_404(AnalysisResult, id=analysis_id)

    # Ensure the user has permission to download this data
    if analysis.jira_file.user != request.user:
        messages.error(request, "You don't have permission to download this data.")
        return redirect('dashboard')

    # Path to the CSV file
    file_path = os.path.join(settings.BASE_DIR, 'cleaned_jira_data_with_impact.csv')

    # Check if file exists
    if not os.path.exists(file_path):
        messages.error(request, "The cleaned data file could not be found.")
        return redirect('view_analysis', analysis_id=analysis_id)

    # Serve the file for download
    response = FileResponse(open(file_path, 'rb'), as_attachment=True, filename='cleaned_jira_data_with_impact.csv')
    return response

@login_required
def delete_file(request, file_id):
    """Delete a JIRA file and its associated analysis results"""
    jira_file = get_object_or_404(JiraFile, id=file_id, user=request.user)

    # Get the file name for the success message
    file_name = jira_file.file.name

    try:
        # Delete the file and all associated analysis results
        jira_file.delete()
        messages.success(request, f"File '{file_name}' has been deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting file: {str(e)}")

    return redirect('dashboard')



@login_required
def client_overview(request):
    """View for the client overview page showing all clients and their metrics over time"""

    # Get all analysis results for the user, ordered by analysis_date (newest first), fallback to creation date
    analysis_results = AnalysisResult.objects.filter(
        jira_file__user=request.user,
        jira_file__processed=True  # Only include processed files
    ).order_by('-jira_file__analysis_date', '-created_at')

    # Initialize data structures to store client metrics over time
    clients_data = defaultdict(list)
    all_clients = set()
    timeline_dates = []

    # Process each analysis result to extract client metrics over time
    for analysis in analysis_results:
        if not analysis.client_metrics:
            continue

        # Use analysis_date if available, otherwise fall back to created_at
        if analysis.jira_file.analysis_date:
            analysis_date = analysis.jira_file.analysis_date.strftime('%Y-%m-%d')
        else:
            analysis_date = analysis.created_at.strftime('%Y-%m-%d')

        if analysis_date not in timeline_dates:
            timeline_dates.append(analysis_date)

        # Process each client's metrics
        for client_name, metrics in analysis.client_metrics.items():
            all_clients.add(client_name)

            # Store the metrics with the date, handling both Client_Impact and Customer_Experience_Score
            client_impact = metrics.get('Client_Impact', 0)
            if client_impact == 0:
                client_impact = metrics.get('Customer_Experience_Score', 0)

            clients_data[client_name].append({
                'date': analysis_date,
                'sentiment': metrics.get('sentiment', 0),
                'priority_impact': metrics.get('Priority_Impact', 0),
                'issue_type_impact': metrics.get('Issue_Type_Impact', 0),
                'tickets': metrics.get('Tickets', 0),
                'resolution_time': metrics.get('Avg_Resolution_Time_Days', 0),
                'client_impact': client_impact,
                'analysis_id': analysis.id
            })

    # Sort timeline dates chronologically
    timeline_dates.sort()

    # Get the latest metrics for each client for the summary view
    latest_metrics = {}
    for client in all_clients:
        if clients_data[client]:
            # Sort by date (newest first) and take the first item
            sorted_data = sorted(clients_data[client], key=lambda x: x['date'], reverse=True)
            latest_metrics[client] = sorted_data[0]

    # Calculate trend data for each client
    client_trends = {}
    for client in all_clients:
        if len(clients_data[client]) > 1:
            # Sort by date (oldest first)
            sorted_data = sorted(clients_data[client], key=lambda x: x['date'])

            # Calculate trends
            first_metrics = sorted_data[0]
            last_metrics = sorted_data[-1]

            client_trends[client] = {
                'sentiment_change': last_metrics['sentiment'] - first_metrics['sentiment'],
                'resolution_time_change': last_metrics['resolution_time'] - first_metrics['resolution_time'],
                'client_impact_change': last_metrics['client_impact'] - first_metrics['client_impact'],
                'tickets_change': last_metrics['tickets'] - first_metrics['tickets'],
                'data_points': len(sorted_data)
            }

    # Feature 3: Sort clients by most recent update date (newest first)
    sorted_clients = []
    for client in all_clients:
        if client in latest_metrics:
            # Get the most recent date for this client
            latest_date = latest_metrics[client].get('date', '1900-01-01')
            sorted_clients.append((client, latest_date))

    # Sort by date (newest first)
    sorted_clients.sort(key=lambda x: x[1], reverse=True)

    # Extract just the client names in the sorted order
    sorted_client_names = [client[0] for client in sorted_clients]

    # Calculate summary metrics
    total_resolution_time = 0
    total_client_impact = 0
    client_count = 0

    # Calculate total tickets by summing issue_count from all processed files
    # This ensures we count all tickets across all files
    processed_analysis_results = analysis_results.filter(jira_file__processed=True)
    total_tickets = sum(analysis.issue_count for analysis in processed_analysis_results)

    for client in all_clients:
        if client in latest_metrics:
            metrics = latest_metrics[client]
            total_resolution_time += metrics.get('resolution_time', 0)
            total_client_impact += metrics.get('client_impact', 0)
            client_count += 1

    # Calculate averages
    avg_resolution_time = total_resolution_time / client_count if client_count > 0 else 0
    avg_client_impact = total_client_impact / client_count if client_count > 0 else 0

    context = {
        'all_clients': sorted_client_names,
        'latest_metrics': latest_metrics,
        'client_trends': client_trends,
        'timeline_dates': timeline_dates,
        'clients_data_json': json.dumps(clients_data),
        'has_data': len(all_clients) > 0,
        'total_tickets': total_tickets,
        'avg_resolution_time': avg_resolution_time,
        'avg_client_impact': avg_client_impact
    }

    return render(request, 'analyzer/client_overview.html', context)

@login_required
def team_overview(request):
    """View for the team overview page showing team performance metrics"""

    # Handle file upload
    if request.method == 'POST':
        form = SurveyFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            survey_file = form.save(commit=False)
            survey_file.user = request.user

            try:
                survey_file.save()

                # Validate file structure
                is_valid, error_message, column_info = validate_survey_file_structure(survey_file.file.path)

                if not is_valid:
                    messages.error(request, f"File validation failed: {error_message}")
                    if column_info.get('missing_columns'):
                        messages.error(request, f"Missing columns: {', '.join(column_info['missing_columns'])}")
                    survey_file.delete()
                    return redirect('team_overview')

                # Process the survey file
                analysis_results = process_survey_file(survey_file)

                # Add detailed success message
                success_msg = f"Survey file '{survey_file.get_filename()}' uploaded and processed successfully! "
                success_msg += f"Processed {analysis_results['total_responses']} total responses "
                success_msg += f"({analysis_results['valid_responses']} valid for KPI calculations)."
                messages.success(request, success_msg)

                # Log the successful processing
                import logging
                logging.info(f"Survey file processed successfully for user {request.user.username}: {survey_file.get_filename()}")

            except Exception as e:
                # Clean up the file if processing failed
                if survey_file.id:
                    survey_file.delete()

                # Provide detailed error message
                error_msg = f"Error processing survey file: {str(e)}"
                messages.error(request, error_msg)

                # Log the error for debugging
                import logging
                import traceback
                logging.error(f"Survey file processing failed for user {request.user.username}: {error_msg}")
                logging.error(traceback.format_exc())

                return redirect('team_overview')

            return redirect('team_overview')
        else:
            # Form validation failed
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = SurveyFileUploadForm()

    # Get survey data for the current user
    survey_files = SurveyFile.objects.filter(user=request.user, processed=True).order_by('-uploaded_at')

    # Get the latest analysis
    latest_analysis = None
    survey_data = None
    kpi_data = None
    feedback_data = None

    if survey_files.exists():
        latest_file = survey_files.first()
        try:
            latest_analysis = latest_file.analysis

            # Prepare data for charts
            kpi_data = {
                'psychological_safety': latest_analysis.avg_psychological_safety or 0,
                'work_environment': latest_analysis.avg_work_environment or 0,
                'client_service': latest_analysis.avg_client_service or 0,
                'team_collaboration': latest_analysis.avg_team_collaboration or 0,
                'overall_satisfaction': latest_analysis.overall_satisfaction or 0,
                'question_averages': latest_analysis.question_averages or {},
                'satisfaction_distribution': latest_analysis.satisfaction_distribution or {},
                'role_distribution': latest_analysis.role_distribution or {}
            }

            # Prepare feedback data with individual responses
            feedback_responses = latest_file.responses.exclude(open_feedback__isnull=True).exclude(open_feedback__exact='').order_by('-created_at')

            # Get unique roles for filtering
            feedback_roles = list(feedback_responses.values_list('role', flat=True).distinct())

            feedback_data = {
                'themes': latest_analysis.feedback_themes or [],
                'total_feedback': latest_analysis.feedback_count or 0,
                'responses': feedback_responses,
                'roles': feedback_roles
            }

            survey_data = {
                'total_responses': latest_analysis.total_responses,
                'valid_responses': latest_analysis.valid_responses,
                'file_name': latest_file.get_filename(),
                'upload_date': latest_file.uploaded_at,
                'survey_date': latest_file.survey_date
            }

        except SurveyAnalysis.DoesNotExist:
            messages.warning(request, "Survey analysis not found. Please re-upload the file.")

    context = {
        'form': form,
        'survey_files': survey_files,
        'latest_analysis': latest_analysis,
        'survey_data': survey_data,
        'kpi_data': kpi_data,
        'feedback_data': feedback_data,
        'has_data': latest_analysis is not None
    }

    return render(request, 'analyzer/team_overview.html', context)

@login_required
def client_detail(request, client_name):
    """View for detailed metrics of a specific client over time"""

    # URL decode the client name
    client_name = urllib.parse.unquote(client_name)

    # Get all analysis results for the user, ordered by analysis_date (newest first), fallback to creation date
    analysis_results = AnalysisResult.objects.filter(
        jira_file__user=request.user,
        jira_file__processed=True  # Only include processed files
    ).order_by('-jira_file__analysis_date', '-created_at')

    # Initialize data structures
    client_metrics_over_time = []
    timeline_dates = []
    client_ticket_types = {}
    latest_analysis = None
    total_client_tickets = 0

    # Process each analysis result to extract this client's metrics over time
    for analysis in analysis_results:
        # Skip if analysis has no client metrics
        if not analysis.client_metrics:
            continue

        # If this client isn't in the current analysis but we have other analyses,
        # we'll still process the file to get ticket type distributions
        has_client_data = client_name in analysis.client_metrics

        # Store the latest analysis that has data (even if not for this specific client)
        if latest_analysis is None:
            latest_analysis = analysis

        # If this analysis has data for our client, process it
        if has_client_data:
            metrics = analysis.client_metrics[client_name]

            # Use analysis_date if available, otherwise fall back to created_at
            if analysis.jira_file.analysis_date:
                analysis_date = analysis.jira_file.analysis_date.strftime('%Y-%m-%d')
            else:
                analysis_date = analysis.created_at.strftime('%Y-%m-%d')

            # Add the date to our timeline
            if analysis_date not in timeline_dates:
                timeline_dates.append(analysis_date)

            # Get ticket count for this client in this analysis
            client_tickets_in_analysis = int(metrics.get('Tickets', 0))
            total_client_tickets += client_tickets_in_analysis

            # Store the metrics with the date
            client_metrics_over_time.append({
                'date': analysis_date,
                'sentiment': metrics.get('sentiment', 0),
                'priority_impact': metrics.get('Priority_Impact', 0),
                'issue_type_impact': metrics.get('Issue_Type_Impact', 0),
                'tickets': client_tickets_in_analysis,
                'resolution_time': metrics.get('Avg_Resolution_Time_Days', 0),
                'client_impact': metrics.get('Client_Impact', 0) if 'Client_Impact' in metrics else metrics.get('Customer_Experience_Score', 0),
                'analysis_id': analysis.id,
                'file_name': analysis.jira_file.file.name
            })

            # No need to track client_ticket_count separately anymore

    # Sort timeline dates chronologically
    timeline_dates.sort()

    # Sort metrics by date (oldest first for trend analysis)
    client_metrics_over_time = sorted(client_metrics_over_time, key=lambda x: x['date'])

    # Calculate trends if we have enough data points
    trends = None
    if len(client_metrics_over_time) > 1:
        first_metrics = client_metrics_over_time[0]
        last_metrics = client_metrics_over_time[-1]

        trends = {
            'sentiment_change': last_metrics['sentiment'] - first_metrics['sentiment'],
            'resolution_time_change': last_metrics['resolution_time'] - first_metrics['resolution_time'],
            'client_impact_change': last_metrics['client_impact'] - first_metrics['client_impact'],
            'data_points': len(client_metrics_over_time)
        }

    # Aggregate ticket types across all analyses for this client
    excluded_prefixes = ['generated at', 'created at', 'updated at', 'date']
    excluded_keywords = ['timestamp', 'time', 'date', 'utc', 'gmt']

    # Initialize combined ticket types dictionary
    client_ticket_types = {}

    # Make sure we have the total tickets for this client
    if not total_client_tickets:
        total_client_tickets = sum(metric['tickets'] for metric in client_metrics_over_time)

    # Process each analysis result to extract and combine ticket types
    for analysis in analysis_results:
        if not analysis.client_metrics or client_name not in analysis.client_metrics:
            continue

        # Get client's ticket count in this analysis
        client_tickets_in_analysis = int(analysis.client_metrics[client_name].get('Tickets', 0))

        if client_tickets_in_analysis > 0 and analysis.ticket_types:
            # Calculate what portion of the total tickets this analysis represents
            analysis_weight = client_tickets_in_analysis / total_client_tickets if total_client_tickets > 0 else 0

            # Process each ticket type in this analysis
            for ticket_type, count in analysis.ticket_types.items():
                # Skip entries that look like timestamps or dates
                if any(ticket_type.lower().startswith(prefix) for prefix in excluded_prefixes):
                    continue
                if any(keyword in ticket_type.lower() for keyword in excluded_keywords):
                    continue

                # Calculate this client's portion of this ticket type
                # Scale by the client's proportion of tickets in this analysis
                client_portion = round(int(count) * analysis_weight)

                # Add to the combined ticket types
                if client_portion > 0:
                    if ticket_type in client_ticket_types:
                        client_ticket_types[ticket_type] += client_portion
                    else:
                        client_ticket_types[ticket_type] = client_portion

    # If no ticket types were found, use a default
    if not client_ticket_types and total_client_tickets > 0:
        client_ticket_types = {"No categorized tickets": total_client_tickets}

    # Ensure the total of ticket types matches the total client tickets
    current_total = sum(client_ticket_types.values())
    if current_total != total_client_tickets and client_ticket_types and total_client_tickets > 0:
        # Find the largest category to adjust
        largest_category = max(client_ticket_types.items(), key=lambda x: x[1])[0]
        # Adjust it to make the total match
        adjustment = total_client_tickets - current_total
        client_ticket_types[largest_category] += adjustment

    context = {
        'client_name': client_name,
        'metrics_over_time': client_metrics_over_time,
        'timeline_dates': timeline_dates,
        'trends': trends,
        'metrics_json': json.dumps(client_metrics_over_time),
        'ticket_types_json': json.dumps(client_ticket_types),
        'has_data': len(client_metrics_over_time) > 0
    }

    return render(request, 'analyzer/client_detail.html', context)


# Export Views for Feature 1
@login_required
def export_csv(request):
    """Export current dashboard data as CSV"""
    # Get the same data as dashboard
    analysis_results = AnalysisResult.objects.filter(
        jira_file__user=request.user,
        jira_file__processed=True
    ).order_by('-jira_file__analysis_date', '-created_at')

    # Prepare data for CSV export
    csv_data = []
    for analysis in analysis_results:
        if not analysis.client_metrics:
            continue

        analysis_date = analysis.jira_file.analysis_date or analysis.created_at.date()

        for client_name, metrics in analysis.client_metrics.items():
            csv_data.append({
                'Client': client_name,
                'Analysis Date': analysis_date,
                'Sentiment': metrics.get('sentiment', 0),
                'Resolution Time (Days)': metrics.get('Avg_Resolution_Time_Days', 0),
                'Client Impact Score': metrics.get('Client_Impact', 0),
                'Tickets': metrics.get('Tickets', 0),
                'File Name': analysis.jira_file.original_filename or analysis.jira_file.file.name
            })

    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="client_metrics_export.csv"'

    if csv_data:
        fieldnames = csv_data[0].keys()
        writer = csv.DictWriter(response, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    return response


@login_required
def export_excel(request):
    """Export current dashboard data as Excel"""
    import pandas as pd

    # Get the same data as dashboard
    analysis_results = AnalysisResult.objects.filter(
        jira_file__user=request.user,
        jira_file__processed=True
    ).order_by('-jira_file__analysis_date', '-created_at')

    # Prepare data for Excel export
    excel_data = []
    for analysis in analysis_results:
        if not analysis.client_metrics:
            continue

        analysis_date = analysis.jira_file.analysis_date or analysis.created_at.date()

        for client_name, metrics in analysis.client_metrics.items():
            excel_data.append({
                'Client': client_name,
                'Analysis Date': analysis_date,
                'Sentiment': metrics.get('sentiment', 0),
                'Resolution Time (Days)': metrics.get('Avg_Resolution_Time_Days', 0),
                'Client Impact Score': metrics.get('Client_Impact', 0),
                'Tickets': metrics.get('Tickets', 0),
                'File Name': analysis.jira_file.original_filename or analysis.jira_file.file.name
            })

    # Create Excel file in memory
    output = BytesIO()
    if excel_data:
        df = pd.DataFrame(excel_data)
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Client Metrics', index=False)

    output.seek(0)
    response = HttpResponse(
        output.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename="client_metrics_export.xlsx"'

    return response


@login_required
def export_pdf(request):
    """Export comprehensive dashboard data as PDF report with visual insights"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        # Fallback if reportlab is not installed
        return HttpResponse("PDF export requires reportlab package to be installed.", status=500)

    # Get comprehensive dashboard data
    analysis_results = AnalysisResult.objects.filter(
        jira_file__user=request.user,
        jira_file__processed=True
    ).order_by('-jira_file__analysis_date', '-created_at')

    # Prepare comprehensive data
    all_clients = set()
    latest_metrics = {}
    total_tickets = 0
    sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
    resolution_times = []
    client_impacts = []

    for analysis in analysis_results:
        if not analysis.client_metrics:
            continue

        total_tickets += analysis.issue_count

        for client_name, metrics in analysis.client_metrics.items():
            all_clients.add(client_name)
            if client_name not in latest_metrics:
                latest_metrics[client_name] = metrics

                # Collect data for statistics
                sentiment = metrics.get('sentiment', 0)
                if sentiment > 0.1:
                    sentiment_distribution['positive'] += 1
                elif sentiment < -0.1:
                    sentiment_distribution['negative'] += 1
                else:
                    sentiment_distribution['neutral'] += 1

                resolution_times.append(metrics.get('Avg_Resolution_Time_Days', 0))
                client_impacts.append(metrics.get('Client_Impact', 0))

    # Calculate summary statistics
    avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    avg_client_impact = sum(client_impacts) / len(client_impacts) if client_impacts else 0
    critical_clients = sum(1 for impact in client_impacts if impact > 0.6)
    excellent_clients = sum(1 for impact in client_impacts if impact < 0.2)

    # Create comprehensive PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Container for the 'Flowable' objects
    elements = []

    # Define comprehensive styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#e31937')
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#333333')
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.HexColor('#495057')
    )

    # Title and header
    title = Paragraph("Comprehensive Client Experience Analysis Report", title_style)
    elements.append(title)

    # Report metadata
    report_date = timezone.now().strftime('%B %d, %Y at %H:%M')
    metadata = Paragraph(f"Generated on {report_date}", styles['Normal'])
    elements.append(metadata)
    elements.append(Spacer(1, 20))

    # Executive Summary with KPIs
    summary_title = Paragraph("📊 Executive Dashboard Summary", heading_style)
    elements.append(summary_title)

    # Enhanced KPI summary with visual indicators
    def get_sentiment_emoji(sentiment):
        if sentiment > 0.1:
            return "😊 Positive"
        elif sentiment < -0.1:
            return "😞 Negative"
        else:
            return "😐 Neutral"

    def get_performance_indicator(value, thresholds):
        if value <= thresholds[0]:
            return "🟢 Excellent"
        elif value <= thresholds[1]:
            return "🟡 Good"
        elif value <= thresholds[2]:
            return "🟠 Fair"
        else:
            return "🔴 Critical"

    summary_data = [
        ['Key Performance Indicator', 'Value', 'Status'],
        ['Total Clients Analyzed', str(len(all_clients)), '📈 Active'],
        ['Total Support Tickets', f"{total_tickets:,}", '🎫 Processed'],
        ['Average Resolution Time', f"{avg_resolution_time:.1f} days", get_performance_indicator(avg_resolution_time, [2, 5, 10])],
        ['Average Experience Score', f"{avg_client_impact:.2f}", get_performance_indicator(avg_client_impact, [0.2, 0.4, 0.6])],
        ['Critical Clients (>0.6 score)', str(critical_clients), '⚠️ Attention Needed' if critical_clients > 0 else '✅ None'],
        ['Excellent Clients (<0.2 score)', str(excellent_clients), '🌟 High Satisfaction'],
        ['Positive Sentiment Clients', str(sentiment_distribution['positive']), '😊 Happy'],
        ['Negative Sentiment Clients', str(sentiment_distribution['negative']), '😞 Needs Attention'],
        ['Neutral Sentiment Clients', str(sentiment_distribution['neutral']), '😐 Stable']
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 1.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e31937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 25))

    # Detailed Client Performance Analysis
    clients_title = Paragraph("🎯 Client Performance Analysis", heading_style)
    elements.append(clients_title)

    # Performance insights paragraph
    insights_text = f"""
    This section provides detailed insights into client performance metrics. Out of {len(all_clients)} clients analyzed:
    • {critical_clients} clients require immediate attention (Critical status)
    • {excellent_clients} clients show excellent satisfaction levels
    • {sentiment_distribution['negative']} clients have negative sentiment requiring intervention
    • Average resolution time across all clients is {avg_resolution_time:.1f} days
    """

    insights_para = Paragraph(insights_text, styles['Normal'])
    elements.append(insights_para)
    elements.append(Spacer(1, 15))

    # Enhanced client data table with visual indicators
    client_data = [['Client Name', 'Sentiment Status', 'Resolution Time', 'Experience Score', 'Tickets', 'Priority']]

    # Sort clients by impact score (worst first) and take top 15
    sorted_clients = sorted(
        latest_metrics.items(),
        key=lambda x: x[1].get('Client_Impact', 0),
        reverse=True
    )[:15]

    for client_name, metrics in sorted_clients:
        sentiment = metrics.get('sentiment', 0)
        resolution_time = metrics.get('Avg_Resolution_Time_Days', 0)
        impact_score = metrics.get('Client_Impact', 0)
        tickets = metrics.get('Tickets', 0)

        # Enhanced sentiment with emojis
        if sentiment > 0.3:
            sentiment_text = f"😊 Positive ({sentiment:.2f})"
        elif sentiment >= 0:
            sentiment_text = f"😐 Neutral ({sentiment:.2f})"
        elif sentiment >= -0.2:
            sentiment_text = f"😕 Uncomfortable ({sentiment:.2f})"
        elif sentiment >= -0.5:
            sentiment_text = f"😠 Annoyed ({sentiment:.2f})"
        elif sentiment >= -0.8:
            sentiment_text = f"😡 Frustrated ({sentiment:.2f})"
        else:
            sentiment_text = f"😵 Very Frustrated ({sentiment:.2f})"

        # Priority based on combined factors
        if impact_score > 0.6 or sentiment < -0.5:
            priority = "🔴 High"
        elif impact_score > 0.4 or sentiment < -0.2:
            priority = "🟡 Medium"
        else:
            priority = "🟢 Low"

        client_data.append([
            client_name[:20] + "..." if len(client_name) > 20 else client_name,
            sentiment_text,
            f"{resolution_time:.1f}d",
            f"{impact_score:.2f}",
            str(int(tickets)),
            priority
        ])

    client_table = Table(client_data, colWidths=[1.8*inch, 1.8*inch, 0.8*inch, 0.8*inch, 0.6*inch, 0.8*inch])
    client_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e31937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(client_table)
    elements.append(Spacer(1, 20))

    # Recommendations section
    recommendations_title = Paragraph("💡 Strategic Recommendations", heading_style)
    elements.append(recommendations_title)

    recommendations_text = f"""
    Based on the analysis of {len(all_clients)} clients and {total_tickets:,} tickets, here are key recommendations:

    🔴 Immediate Actions Required:
    • Focus on {critical_clients} critical clients with experience scores > 0.6
    • Address {sentiment_distribution['negative']} clients with negative sentiment
    • Prioritize clients with resolution times > {avg_resolution_time + 5:.1f} days

    🟡 Medium-term Improvements:
    • Implement proactive monitoring for clients trending toward negative sentiment
    • Optimize support processes to reduce average resolution time below {avg_resolution_time:.1f} days
    • Develop client success programs for maintaining {excellent_clients} excellent-performing clients

    🟢 Long-term Strategy:
    • Establish benchmarks based on top-performing client metrics
    • Create early warning systems for sentiment degradation
    • Implement regular client health check processes
    """

    recommendations_para = Paragraph(recommendations_text, styles['Normal'])
    elements.append(recommendations_para)

    # Build PDF
    doc.build(elements)

    # Get the value of the BytesIO buffer and write it to the response
    pdf = buffer.getvalue()
    buffer.close()

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="client_analysis_report.pdf"'
    response.write(pdf)

    return response


# Survey Export Views
@login_required
def export_survey_csv(request):
    """Export current survey data as CSV"""
    # Get the latest survey file for the user
    survey_file = SurveyFile.objects.filter(user=request.user, processed=True).order_by('-uploaded_at').first()

    if not survey_file:
        messages.error(request, "No survey data available for export.")
        return redirect('team_overview')

    # Prepare data for CSV export
    csv_data = []
    for response in survey_file.responses.all():
        csv_data.append({
            'Response ID': str(response.response_id),
            'Role': response.role,
            'Q1.1 Speaking Up': response.q1_1_speaking_up,
            'Q1.2 Mistakes Held Against': response.q1_2_mistakes_held_against,
            'Q1.3 Respect When Not Knowing': response.q1_3_respect_when_not_knowing,
            'Q2.1 Workload Manageable': response.q2_1_workload_manageable,
            'Q2.2 Tools and Resources': response.q2_2_tools_and_resources,
            'Q2.3 Work Life Balance': response.q2_3_work_life_balance,
            'Q3.1 Understanding Clients': response.q3_1_understanding_clients,
            'Q3.2 Support Handling Clients': response.q3_2_support_handling_clients,
            'Q3.3 Tools for Client Service': response.q3_3_tools_for_client_service,
            'Q4.1 Help Responsiveness': response.q4_1_help_responsiveness,
            'Q4.2 Conflict Resolution': response.q4_2_conflict_resolution,
            'Q4.3 Sharing Updates': response.q4_3_sharing_updates,
            'Open Feedback': response.open_feedback,
            'Created At': response.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="survey_responses_export.csv"'

    if csv_data:
        fieldnames = csv_data[0].keys()
        writer = csv.DictWriter(response, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    return response


@login_required
def export_survey_excel(request):
    """Export current survey data as Excel"""
    import pandas as pd

    # Get the latest survey file for the user
    survey_file = SurveyFile.objects.filter(user=request.user, processed=True).order_by('-uploaded_at').first()

    if not survey_file:
        messages.error(request, "No survey data available for export.")
        return redirect('team_overview')

    # Prepare data for Excel export
    excel_data = []
    for response in survey_file.responses.all():
        excel_data.append({
            'Response ID': str(response.response_id),
            'Role': response.role,
            'Q1.1 Speaking Up': response.q1_1_speaking_up,
            'Q1.2 Mistakes Held Against': response.q1_2_mistakes_held_against,
            'Q1.3 Respect When Not Knowing': response.q1_3_respect_when_not_knowing,
            'Q2.1 Workload Manageable': response.q2_1_workload_manageable,
            'Q2.2 Tools and Resources': response.q2_2_tools_and_resources,
            'Q2.3 Work Life Balance': response.q2_3_work_life_balance,
            'Q3.1 Understanding Clients': response.q3_1_understanding_clients,
            'Q3.2 Support Handling Clients': response.q3_2_support_handling_clients,
            'Q3.3 Tools for Client Service': response.q3_3_tools_for_client_service,
            'Q4.1 Help Responsiveness': response.q4_1_help_responsiveness,
            'Q4.2 Conflict Resolution': response.q4_2_conflict_resolution,
            'Q4.3 Sharing Updates': response.q4_3_sharing_updates,
            'Open Feedback': response.open_feedback,
            'Created At': response.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    # Create Excel file in memory
    output = BytesIO()
    if excel_data:
        df = pd.DataFrame(excel_data)

        # Also include analysis summary
        try:
            analysis = survey_file.analysis
            summary_data = [{
                'Metric': 'Total Responses',
                'Value': analysis.total_responses
            }, {
                'Metric': 'Valid Responses (for KPIs)',
                'Value': analysis.valid_responses
            }, {
                'Metric': 'Overall Satisfaction',
                'Value': f"{analysis.overall_satisfaction:.2f}/5" if analysis.overall_satisfaction else "N/A"
            }, {
                'Metric': 'Psychological Safety',
                'Value': f"{analysis.avg_psychological_safety:.2f}/5" if analysis.avg_psychological_safety else "N/A"
            }, {
                'Metric': 'Work Environment',
                'Value': f"{analysis.avg_work_environment:.2f}/5" if analysis.avg_work_environment else "N/A"
            }, {
                'Metric': 'Client Service',
                'Value': f"{analysis.avg_client_service:.2f}/5" if analysis.avg_client_service else "N/A"
            }, {
                'Metric': 'Team Collaboration',
                'Value': f"{analysis.avg_team_collaboration:.2f}/5" if analysis.avg_team_collaboration else "N/A"
            }]

            summary_df = pd.DataFrame(summary_data)

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Survey Responses', index=False)
                summary_df.to_excel(writer, sheet_name='Analysis Summary', index=False)
        except SurveyAnalysis.DoesNotExist:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Survey Responses', index=False)

    output.seek(0)
    response = HttpResponse(
        output.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename="survey_responses_export.xlsx"'

    return response


@login_required
def export_team_overview_pdf(request):
    """Export comprehensive team overview data as PDF report"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        # Fallback if reportlab is not installed
        return HttpResponse("PDF export requires reportlab package to be installed.", status=500)

    # Get the latest survey file for the user
    survey_file = SurveyFile.objects.filter(user=request.user, processed=True).order_by('-uploaded_at').first()

    if not survey_file:
        messages.error(request, "No survey data available for export.")
        return redirect('team_overview')

    try:
        analysis = survey_file.analysis
    except SurveyAnalysis.DoesNotExist:
        messages.error(request, "No analysis data available for export.")
        return redirect('team_overview')

    # Create comprehensive PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Container for the 'Flowable' objects
    elements = []

    # Define comprehensive styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#e31937')
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#333333')
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.HexColor('#495057')
    )

    # Title and header
    title = Paragraph("Team Overview Analysis Report", title_style)
    elements.append(title)

    # Report metadata
    report_date = timezone.now().strftime('%B %d, %Y at %H:%M')
    metadata = Paragraph(f"Generated on {report_date}", styles['Normal'])
    elements.append(metadata)

    # Survey file info
    file_info = Paragraph(f"Survey File: {survey_file.get_filename()}", styles['Normal'])
    elements.append(file_info)
    if survey_file.survey_date:
        survey_date_info = Paragraph(f"Survey Date: {survey_file.survey_date.strftime('%B %d, %Y')}", styles['Normal'])
        elements.append(survey_date_info)

    elements.append(Spacer(1, 20))

    # Executive Summary with KPIs
    summary_title = Paragraph("📊 Executive Summary", heading_style)
    elements.append(summary_title)

    # Enhanced KPI summary
    def get_satisfaction_level(score):
        if score >= 4.0:
            return "🟢 High"
        elif score >= 3.0:
            return "🟡 Medium"
        else:
            return "🔴 Low"

    summary_data = [
        ['Key Performance Indicator', 'Score', 'Level'],
        ['Overall Team Satisfaction', f"{analysis.overall_satisfaction:.2f}/5" if analysis.overall_satisfaction else "N/A",
         get_satisfaction_level(analysis.overall_satisfaction) if analysis.overall_satisfaction else "N/A"],
        ['Psychological Safety', f"{analysis.avg_psychological_safety:.2f}/5" if analysis.avg_psychological_safety else "N/A",
         get_satisfaction_level(analysis.avg_psychological_safety) if analysis.avg_psychological_safety else "N/A"],
        ['Work Environment', f"{analysis.avg_work_environment:.2f}/5" if analysis.avg_work_environment else "N/A",
         get_satisfaction_level(analysis.avg_work_environment) if analysis.avg_work_environment else "N/A"],
        ['Client Service', f"{analysis.avg_client_service:.2f}/5" if analysis.avg_client_service else "N/A",
         get_satisfaction_level(analysis.avg_client_service) if analysis.avg_client_service else "N/A"],
        ['Team Collaboration', f"{analysis.avg_team_collaboration:.2f}/5" if analysis.avg_team_collaboration else "N/A",
         get_satisfaction_level(analysis.avg_team_collaboration) if analysis.avg_team_collaboration else "N/A"],
        ['Total Survey Responses', str(analysis.total_responses), '📈 Collected'],
        ['Valid Responses for KPIs', str(analysis.valid_responses), '✅ Analyzed'],
        ['Response Rate for KPIs', f"{(analysis.valid_responses/analysis.total_responses*100):.1f}%" if analysis.total_responses > 0 else "0%", '📊 Coverage']
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 1.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e31937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 25))

    # Role Distribution Analysis
    roles_title = Paragraph("👥 Team Role Distribution", heading_style)
    elements.append(roles_title)

    role_data = [['Role', 'Count', 'Percentage']]
    total_roles = sum(analysis.role_distribution.values()) if analysis.role_distribution else 0

    for role, count in (analysis.role_distribution or {}).items():
        percentage = (count / total_roles * 100) if total_roles > 0 else 0
        role_data.append([role, str(count), f"{percentage:.1f}%"])

    if len(role_data) > 1:
        role_table = Table(role_data, colWidths=[3*inch, 1*inch, 1*inch])
        role_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e31937')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(role_table)
    else:
        elements.append(Paragraph("No role distribution data available.", styles['Normal']))

    elements.append(Spacer(1, 20))

    # Satisfaction Distribution
    satisfaction_title = Paragraph("📈 Satisfaction Level Distribution", heading_style)
    elements.append(satisfaction_title)

    satisfaction_dist = analysis.satisfaction_distribution or {}
    satisfaction_data = [
        ['Satisfaction Level', 'Count', 'Description'],
        ['High (4.0-5.0)', str(satisfaction_dist.get('high', 0)), '😊 Very Satisfied'],
        ['Medium (3.0-3.9)', str(satisfaction_dist.get('medium', 0)), '😐 Moderately Satisfied'],
        ['Low (1.0-2.9)', str(satisfaction_dist.get('low', 0)), '😞 Needs Improvement']
    ]

    satisfaction_table = Table(satisfaction_data, colWidths=[2*inch, 1*inch, 3*inch])
    satisfaction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e31937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(satisfaction_table)
    elements.append(Spacer(1, 25))

    # Feedback Themes Analysis
    feedback_title = Paragraph("💬 Key Feedback Themes", heading_style)
    elements.append(feedback_title)

    if analysis.feedback_themes:
        feedback_text = f"Analysis of {analysis.feedback_count} open-ended feedback responses revealed the following key themes:\n\n"

        for i, theme in enumerate(analysis.feedback_themes[:10], 1):
            feedback_text += f"{i}. {theme['theme']}: {theme['count']} mentions\n"

        feedback_para = Paragraph(feedback_text, styles['Normal'])
        elements.append(feedback_para)
    else:
        elements.append(Paragraph("No feedback themes available.", styles['Normal']))

    elements.append(Spacer(1, 20))

    # Recommendations section
    recommendations_title = Paragraph("💡 Strategic Recommendations", heading_style)
    elements.append(recommendations_title)

    # Generate recommendations based on scores
    recommendations = []

    if analysis.avg_psychological_safety and analysis.avg_psychological_safety < 3.5:
        recommendations.append("🔴 Psychological Safety: Focus on creating a more open environment where team members feel safe to speak up and make mistakes.")

    if analysis.avg_work_environment and analysis.avg_work_environment < 3.5:
        recommendations.append("🟡 Work Environment: Improve workload management, tools, and work-life balance initiatives.")

    if analysis.avg_client_service and analysis.avg_client_service < 3.5:
        recommendations.append("🔵 Client Service: Enhance client understanding, support processes, and service tools.")

    if analysis.avg_team_collaboration and analysis.avg_team_collaboration < 3.5:
        recommendations.append("🟢 Team Collaboration: Strengthen communication, conflict resolution, and information sharing.")

    if not recommendations:
        recommendations.append("🎉 Excellent Performance: All KPI areas are performing well. Focus on maintaining current standards and continuous improvement.")

    recommendations_text = "\n\n".join(recommendations)
    recommendations_para = Paragraph(recommendations_text, styles['Normal'])
    elements.append(recommendations_para)

    # Build PDF
    doc.build(elements)

    # Get the value of the BytesIO buffer and write it to the response
    pdf = buffer.getvalue()
    buffer.close()

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="team_overview_report.pdf"'
    response.write(pdf)

    return response


# Notes Views for Feature 2
@login_required
def get_client_note(request, client_name):
    """Get the note for a specific client"""
    client_name = urllib.parse.unquote(client_name)

    try:
        note = ClientNote.objects.get(user=request.user, client_name=client_name)
        return JsonResponse({
            'success': True,
            'note_text': note.note_text,
            'updated_at': note.updated_at.strftime('%Y-%m-%d %H:%M')
        })
    except ClientNote.DoesNotExist:
        return JsonResponse({
            'success': True,
            'note_text': '',
            'updated_at': None
        })


@login_required
def save_client_note(request, client_name):
    """Save or update a note for a specific client"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST method allowed'})

    client_name = urllib.parse.unquote(client_name)
    note_text = request.POST.get('note_text', '').strip()

    if not note_text:
        # Delete the note if text is empty
        try:
            note = ClientNote.objects.get(user=request.user, client_name=client_name)
            note.delete()
            return JsonResponse({'success': True, 'message': 'Note deleted'})
        except ClientNote.DoesNotExist:
            return JsonResponse({'success': True, 'message': 'No note to delete'})

    # Create or update the note
    note, created = ClientNote.objects.update_or_create(
        user=request.user,
        client_name=client_name,
        defaults={'note_text': note_text}
    )

    return JsonResponse({
        'success': True,
        'message': 'Note saved successfully',
        'updated_at': note.updated_at.strftime('%Y-%m-%d %H:%M')
    })


# Chatbot API Views
@login_required
@require_http_methods(['POST'])
def chat_message(request):
    """Handle chat messages and return AI responses"""
    try:
        data = json.loads(request.body)
        message = data.get('message')
        session_id = data.get('session_id')

        if not message:
            return JsonResponse({'success': False, 'error': 'Message is required'})

        # Initialize chatbot service
        chatbot = ChatbotService(request.user)
        
        # Process message and get response
        result = chatbot.process_message(message, session_id)
        
        return JsonResponse({
            'success': True,
            'response': result.get('response', ''),
            'session_id': result.get('session_id'),
            'ai_mode': result.get('ai_mode', 'fallback'),
            'rag_metadata': result.get('rag_metadata', {})
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_http_methods(['POST'])
def refresh_rag_index(request):
    """Refresh the RAG index with latest data"""
    try:
        from .chatbot_service import ChatbotService

        chatbot = ChatbotService(request.user)
        result = chatbot.refresh_rag_index()

        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_http_methods(['POST'])
def comprehensive_data_index(request):
    """Perform comprehensive historical data indexing"""
    try:
        from .rag_service import RAGService
        from .document_processors import DocumentProcessorManager
        from .models import AnalysisResult, SurveyAnalysis, ClientNote

        data = json.loads(request.body) if request.body else {}
        force_reindex = data.get('force', False)

        # Initialize services
        rag_service = RAGService(request.user)
        if not rag_service.is_available():
            return JsonResponse({
                'success': False,
                'error': 'RAG service not available. Please ensure ChromaDB is installed.'
            })

        processor_manager = DocumentProcessorManager(request.user)

        # Get data statistics
        stats = {
            'analysis_results': AnalysisResult.objects.filter(
                jira_file__user=request.user,
                jira_file__processed=True
            ).count(),
            'survey_analyses': SurveyAnalysis.objects.filter(
                survey_file__user=request.user,
                survey_file__processed=True
            ).count(),
            'client_notes': ClientNote.objects.filter(user=request.user).count()
        }

        # Process all historical data
        documents = processor_manager.process_all_documents()

        if not documents:
            return JsonResponse({
                'success': False,
                'error': 'No data available for indexing',
                'stats': stats
            })

        # Clear existing index if force reindex
        if force_reindex:
            rag_service.clear_collection()

        # Index documents
        success = rag_service.add_documents(documents)

        if success:
            collection_stats = rag_service.get_collection_stats()

            # Update indexing metadata
            AnalysisResult.objects.filter(
                jira_file__user=request.user,
                jira_file__processed=True
            ).update(rag_indexed=True, rag_indexed_at=timezone.now())

            SurveyAnalysis.objects.filter(
                survey_file__user=request.user,
                survey_file__processed=True
            ).update(rag_indexed=True, rag_indexed_at=timezone.now())

            return JsonResponse({
                'success': True,
                'message': f'Successfully indexed {len(documents)} documents',
                'stats': stats,
                'collection_stats': collection_stats,
                'documents_processed': len(documents)
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to index documents',
                'stats': stats
            })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_http_methods(['GET'])
def rag_stats(request):
    """Get RAG service statistics"""
    try:
        from .chatbot_service import ChatbotService

        chatbot = ChatbotService(request.user)
        stats = chatbot.get_rag_stats()

        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_http_methods(['POST'])
def search_knowledge_base(request):
    """Search the knowledge base directly"""
    try:
        data = json.loads(request.body)
        query = data.get('query')
        n_results = data.get('n_results', 5)

        if not query:
            return JsonResponse({'success': False, 'error': 'Query is required'})

        from .chatbot_service import ChatbotService

        chatbot = ChatbotService(request.user)
        results = chatbot.search_knowledge_base(query, n_results)

        return JsonResponse({
            'success': True,
            'results': results,
            'query': query
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def chat_sessions(request):
    """Get user's chat sessions"""
    try:
        sessions = ChatSession.objects.filter(
            user=request.user,
            is_active=True
        ).order_by('-updated_at')[:10]  # Last 10 sessions

        sessions_data = [
            {
                'session_id': session.session_id,
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'message_count': session.messages.count()
            }
            for session in sessions
        ]

        return JsonResponse({
            'success': True,
            'sessions': sessions_data
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error fetching sessions: {str(e)}'
        })


@login_required
def chat_history(request, session_id):
    """Get conversation history for a specific session"""
    try:
        session = get_object_or_404(ChatSession, session_id=session_id, user=request.user)

        chatbot = ChatbotService(request.user)
        history = chatbot.get_conversation_history(session)

        return JsonResponse({
            'success': True,
            'session_id': session_id,
            'title': session.title,
            'history': history
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error fetching history: {str(e)}'
        })


@login_required
@require_http_methods(["POST"])
def delete_chat_session(request, session_id):
    """Delete a chat session"""
    try:
        session = get_object_or_404(ChatSession, session_id=session_id, user=request.user)
        session.delete()

        return JsonResponse({
            'success': True,
            'message': 'Session deleted successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error deleting session: {str(e)}'
        })


@login_required
def chat_data_summary(request):
    """Get a summary of available data for the chatbot context"""
    try:
        data_service = DataAnalysisService(request.user)
        overview = data_service.get_client_overview()
        performance = data_service.get_performance_metrics()

        return JsonResponse({
            'success': True,
            'data_summary': {
                'overview': overview,
                'performance': performance,
                'has_data': 'error' not in overview
            }
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error fetching data summary: {str(e)}'
        })


@login_required
@require_http_methods(['POST'])
def debug_rag_retrieval(request):
    """Debug endpoint to analyze RAG retrieval for specific queries"""
    try:
        data = json.loads(request.body)
        query = data.get('query', 'client overview')
        n_results = data.get('n_results', 10)

        from .rag_service import RAGService, RAGQueryProcessor

        rag_service = RAGService(request.user)
        if not rag_service.is_available():
            return JsonResponse({
                'success': False,
                'error': 'RAG service not available'
            })

        # Get collection info
        collection_stats = rag_service.get_collection_stats()

        # Test different search strategies
        results = {}

        # 1. Raw semantic search (no filters)
        raw_results = rag_service.search_documents(query, n_results=n_results)
        results['raw_semantic'] = {
            'count': len(raw_results),
            'results': [
                {
                    'id': doc['id'],
                    'distance': doc.get('distance', 'N/A'),
                    'doc_type': doc.get('metadata', {}).get('doc_type', 'Unknown'),
                    'source_type': doc.get('metadata', {}).get('source_type', 'Unknown'),
                    'source': doc.get('metadata', {}).get('source', 'Unknown'),
                    'content_preview': doc.get('content', '')[:200] + '...'
                }
                for doc in raw_results
            ]
        }

        # 2. Filtered search for client data
        client_filter = {
            "doc_type": {"$in": ["client_overview", "client_overview_summary", "client_data"]}
        }
        filtered_results = rag_service.search_documents(query, n_results=n_results, filter_metadata=client_filter)
        results['filtered_client'] = {
            'count': len(filtered_results),
            'filter_used': client_filter,
            'results': [
                {
                    'id': doc['id'],
                    'distance': doc.get('distance', 'N/A'),
                    'doc_type': doc.get('metadata', {}).get('doc_type', 'Unknown'),
                    'source_type': doc.get('metadata', {}).get('source_type', 'Unknown'),
                    'source': doc.get('metadata', {}).get('source', 'Unknown'),
                    'content_preview': doc.get('content', '')[:200] + '...'
                }
                for doc in filtered_results
            ]
        }

        # 3. Query processor results
        query_processor = RAGQueryProcessor(rag_service)
        processed_result = query_processor.process_query(query)
        results['query_processor'] = {
            'query_type': processed_result.get('query_type'),
            'confidence': processed_result.get('confidence'),
            'sources_count': len(processed_result.get('sources', [])),
            'context_length': len(processed_result.get('context', '')),
            'search_method': processed_result.get('search_method', 'unknown')
        }

        # 4. All documents analysis
        try:
            all_docs = rag_service.collection.get(include=["metadatas"])
            doc_type_counts = {}
            source_type_counts = {}

            for metadata in all_docs['metadatas']:
                doc_type = metadata.get('doc_type', 'unknown')
                source_type = metadata.get('source_type', 'unknown')

                doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1

            results['collection_analysis'] = {
                'total_documents': len(all_docs['metadatas']),
                'doc_type_distribution': doc_type_counts,
                'source_type_distribution': source_type_counts
            }
        except Exception as e:
            results['collection_analysis'] = {'error': str(e)}

        return JsonResponse({
            'success': True,
            'query': query,
            'collection_stats': collection_stats,
            'debug_results': results
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
