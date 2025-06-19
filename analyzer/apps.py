from django.apps import AppConfig

class AnalyzerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analyzer'

    def ready(self):
        """Import signals when Django starts"""
        try:
            import analyzer.signals
        except ImportError:
            pass
    
    def ready(self):
        """Initialize app when Django starts."""
        # This ensures templatetags are loaded
        import analyzer.templatetags.analyzer_filters 