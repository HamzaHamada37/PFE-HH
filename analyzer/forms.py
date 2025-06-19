from django import forms
from .models import JiraFile, SurveyFile
from datetime import date, timedelta

class JiraFileUploadForm(forms.ModelForm):
    analysis_date = forms.DateField(
        label='Analysis Date',
        help_text='Select the date for filtering and contextualizing the analysis results',
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'form-control',
            'max': date.today().isoformat(),
            'min': (date.today() - timedelta(days=365*5)).isoformat(),  # 5 years back
        }),
        initial=date.today,
        required=True
    )

    class Meta:
        model = JiraFile
        fields = ['file', 'analysis_date']
        widgets = {
            'file': forms.FileInput(attrs={'class': 'form-control'})
        }

    def clean_analysis_date(self):
        analysis_date = self.cleaned_data.get('analysis_date')
        if analysis_date:
            # Ensure the date is not in the future
            if analysis_date > date.today():
                raise forms.ValidationError("Analysis date cannot be in the future.")

            # Ensure the date is not too far in the past (more than 10 years)
            min_date = date.today() - timedelta(days=365*10)
            if analysis_date < min_date:
                raise forms.ValidationError("Analysis date cannot be more than 10 years in the past.")

        return analysis_date

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            ext = file.name.split('.')[-1].lower()
            if ext not in ['csv', 'xlsx', 'xls']:
                raise forms.ValidationError("Only CSV and Excel files are supported.")

            # Set the file_type field based on the file extension
            if ext == 'csv':
                self.instance.file_type = 'csv'
            else:
                self.instance.file_type = 'xlsx'

            # Save the original filename
            self.instance.original_filename = file.name

        return file


class SurveyFileUploadForm(forms.ModelForm):
    """Form for uploading survey Excel files"""
    survey_date = forms.DateField(
        label='Survey Date',
        help_text='Select the date when the survey was conducted',
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'form-control',
            'max': date.today().isoformat(),
            'min': (date.today() - timedelta(days=365*2)).isoformat(),  # 2 years back
        }),
        initial=date.today,
        required=False
    )

    class Meta:
        model = SurveyFile
        fields = ['file', 'survey_date']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.xlsx,.xls',
                'id': 'surveyFileInput'
            })
        }

    def clean_survey_date(self):
        survey_date = self.cleaned_data.get('survey_date')
        if survey_date:
            # Ensure the date is not in the future
            if survey_date > date.today():
                raise forms.ValidationError("Survey date cannot be in the future.")

            # Ensure the date is not too far in the past (more than 5 years)
            min_date = date.today() - timedelta(days=365*5)
            if survey_date < min_date:
                raise forms.ValidationError("Survey date cannot be more than 5 years in the past.")

        return survey_date

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            ext = file.name.split('.')[-1].lower()
            if ext not in ['xlsx', 'xls']:
                raise forms.ValidationError("Only Excel files (.xlsx, .xls) are supported for survey data.")

            # Check file size (limit to 10MB)
            if file.size > 10 * 1024 * 1024:
                raise forms.ValidationError("File size cannot exceed 10MB.")

            # Set the file_type field based on the file extension
            self.instance.file_type = ext

            # Save the original filename
            self.instance.original_filename = file.name

        return file