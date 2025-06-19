from django.core.management.base import BaseCommand
from analyzer.models import AnalysisResult
import json

class Command(BaseCommand):
    help = 'Inspects client metrics data in the database'

    def handle(self, *args, **options):
        # Get all analysis results
        analysis_results = AnalysisResult.objects.all()
        
        self.stdout.write(f"Found {analysis_results.count()} analysis results")
        
        for i, result in enumerate(analysis_results):
            self.stdout.write(f"\n--- Analysis Result #{i+1} ---")
            self.stdout.write(f"ID: {result.id}")
            self.stdout.write(f"File: {result.jira_file.file.name}")
            self.stdout.write(f"Created: {result.created_at}")
            
            # Print client metrics
            self.stdout.write("\nClient Metrics:")
            if not result.client_metrics:
                self.stdout.write("  No client metrics found!")
                continue
                
            for client_name, metrics in result.client_metrics.items():
                self.stdout.write(f"\n  Client: {client_name}")
                for key, value in metrics.items():
                    self.stdout.write(f"    {key}: {value}")
