from django import template
from django.db.models import QuerySet
import statistics

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Get an item from a dictionary using any key, including keys with spaces.
    This is useful for accessing dictionary items in templates where the key contains spaces.

    Usage: {{ my_dict|get_item:"key with spaces" }}
    """
    if dictionary is None:
        return None

    # Try direct access first
    if key in dictionary:
        return dictionary[key]

    # For pandas DataFrame converted to dict, try string version of the key
    if isinstance(key, str) and key in dictionary:
        return dictionary[key]

    # Return None if key not found
    return None

@register.filter
def count_processed(queryset):
    """Returns the count of processed files in a queryset"""
    return queryset.filter(processed=True).count()

@register.filter
def count_pending(queryset):
    """Returns the count of pending (not processed) files in a queryset"""
    return queryset.filter(processed=False).count()

@register.filter
def first_item(value):
    """Returns the first item in a list or queryset"""
    if isinstance(value, (list, tuple, QuerySet)) and len(value) > 0:
        return value[0]
    return 0

@register.filter
def last_item(value):
    """Returns the last item in a list or queryset"""
    if isinstance(value, (list, tuple, QuerySet)) and len(value) > 0:
        return value[-1]
    return 0

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def map_attribute(dict_list, attr_name):
    """Extract a specific attribute from a list of dictionaries"""
    result = []
    for item in dict_list:
        if attr_name in item:
            result.append(item[attr_name])
    return result

@register.filter
def average(value_list):
    """Calculate the average of a list of values"""
    try:
        if not value_list:
            return 0
        return statistics.mean([float(x) for x in value_list if x is not None])
    except (ValueError, TypeError):
        return 0

@register.filter
def sum_values(value_list):
    """Calculate the sum of a list of values"""
    try:
        return sum([float(x) for x in value_list if x is not None])
    except (ValueError, TypeError):
        return 0

@register.filter
def div(value, arg):
    """Divide the value by the argument"""
    try:
        if arg == 0:
            return 0
        return float(value) / float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def absolute(value):
    """Return the absolute value"""
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0

@register.filter
def split(value, delimiter):
    """Split a string by a delimiter and return the resulting list"""
    if value is None:
        return []
    return value.split(delimiter)

@register.filter
def get_experience_score(metrics_dict):
    """
    Get the Customer Experience Score from a metrics dictionary.
    Tries different field names to handle different data formats.
    """
    if metrics_dict is None:
        return 0.0

    # Try different possible field names
    for field_name in ['Customer_Experience_Score', 'customer_experience_score', 'Client_Impact', 'client_impact', 'urgency_score']:
        if field_name in metrics_dict:
            return metrics_dict[field_name]

    # Default value if no field is found
    return 0.0