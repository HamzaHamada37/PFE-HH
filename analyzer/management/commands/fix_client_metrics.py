from django.core.management.base import BaseCommand
from analyzer.models import AnalysisResult
import json

class Command(BaseCommand):
    help = 'Fixes client metrics data in the database by copying Client_Impact to client_impact'

    def handle(self, *args, **options):
        # Get all analysis results
        analysis_results = AnalysisResult.objects.all()
        
        self.stdout.write(f"Found {analysis_results.count()} analysis results")
        updated_count = 0
        
        for result in analysis_results:
            modified = False
            
            if not result.client_metrics:
                continue
                
            # Create a copy of the client_metrics to modify
            updated_metrics = {}
            
            for client_name, metrics in result.client_metrics.items():
                # Create a copy of the metrics for this client
                updated_client_metrics = dict(metrics)
                
                # If Client_Impact exists, add a lowercase version
                if 'Client_Impact' in metrics:
                    # Add lowercase version for template access
                    updated_client_metrics['client_impact'] = metrics['Client_Impact']
                    modified = True
                
                # Store the updated metrics for this client
                updated_metrics[client_name] = updated_client_metrics
            
            # Update the result if modified
            if modified:
                result.client_metrics = updated_metrics
                result.save()
                updated_count += 1
        
        self.stdout.write(self.style.SUCCESS(f'Successfully updated {updated_count} analysis results'))
