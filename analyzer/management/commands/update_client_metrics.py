from django.core.management.base import BaseCommand
from analyzer.models import AnalysisResult
import json

class Command(BaseCommand):
    help = 'Updates client metrics to ensure Customer_Experience_Score is properly set'

    def handle(self, *args, **options):
        # Get all analysis results
        analysis_results = AnalysisResult.objects.all()
        updated_count = 0

        for result in analysis_results:
            modified = False
            
            # Check if client_metrics is a dictionary
            if isinstance(result.client_metrics, dict):
                # Iterate through each client's metrics
                for client_name, metrics in result.client_metrics.items():
                    # Check if metrics is a dictionary
                    if isinstance(metrics, dict):
                        # If Customer_Experience_Score doesn't exist but client_impact does
                        if 'Customer_Experience_Score' not in metrics and 'client_impact' in metrics:
                            # Copy client_impact to Customer_Experience_Score
                            metrics['Customer_Experience_Score'] = metrics['client_impact']
                            modified = True
                        
                        # If neither exists, check for other potential field names
                        elif 'Customer_Experience_Score' not in metrics and 'client_impact' not in metrics:
                            # Check for other potential field names
                            for field in ['urgency_score', 'impact_score', 'experience_score']:
                                if field in metrics:
                                    metrics['Customer_Experience_Score'] = metrics[field]
                                    modified = True
                                    break
            
            # Save the result if modified
            if modified:
                result.save()
                updated_count += 1
        
        self.stdout.write(self.style.SUCCESS(f'Successfully updated {updated_count} analysis results'))
