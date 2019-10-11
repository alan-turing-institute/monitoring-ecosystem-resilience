"""
WTForms forms for user to enter label
"""
from wtforms import Form, FloatField, FormField, IntegerField, FileField, RadioField, \
    SelectField, validators, FieldList, StringField,SelectMultipleField, HiddenField, widgets



class LabelForm(Form):
    """
    Standard WTForm
    """
    veg_label = RadioField(choices=[("Gaps","Gaps"),
                                     ("Labrynths","Labrynths"),
                                     ("Spots","Spots"),
                                     ("Other","Other")]
                             ,label="Label")
    notes = StringField('Notes:', [validators.optional(), validators.length(max=300)], default=" ")
