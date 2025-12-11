"""
Fake Data Lab routes - Generate synthetic data using Faker library.
"""

import io
import csv
from flask import Blueprint, render_template, request, jsonify, Response
from faker import Faker

data_bp = Blueprint('data', __name__)

# Supported locales
LOCALES = {
    'en_US': 'English (US)',
    'en_GB': 'English (UK)',
    'de_DE': 'German',
    'fr_FR': 'French',
    'es_ES': 'Spanish',
    'it_IT': 'Italian',
    'ja_JP': 'Japanese',
    'zh_CN': 'Chinese',
    'hi_IN': 'Hindi (India)'
}

# Predefined schemas
SCHEMAS = {
    'user_profiles': {
        'name': 'User Profiles',
        'fields': ['name', 'email', 'phone', 'address', 'date_of_birth', 'job']
    },
    'ecommerce_orders': {
        'name': 'E-commerce Orders',
        'fields': ['order_id', 'product_name', 'quantity', 'price', 'customer_name', 'order_date']
    },
    'addresses': {
        'name': 'Addresses',
        'fields': ['street_address', 'city', 'state', 'postal_code', 'country']
    }
}

# Available field types for custom schema
FIELD_TYPES = {
    'name': 'Full Name',
    'first_name': 'First Name',
    'last_name': 'Last Name',
    'email': 'Email Address',
    'phone': 'Phone Number',
    'address': 'Full Address',
    'street_address': 'Street Address',
    'city': 'City',
    'state': 'State/Province',
    'postal_code': 'Postal Code',
    'country': 'Country',
    'company': 'Company Name',
    'job': 'Job Title',
    'date_of_birth': 'Date of Birth',
    'date': 'Random Date',
    'credit_card_masked': 'Credit Card (Masked)',
    'ssn_masked': 'SSN (Masked)',
    'username': 'Username',
    'password': 'Password',
    'url': 'Website URL',
    'ipv4': 'IP Address',
    'uuid': 'UUID',
    'text': 'Random Text',
    'order_id': 'Order ID',
    'product_name': 'Product Name',
    'quantity': 'Quantity',
    'price': 'Price'
}


def generate_field_value(fake: Faker, field_type: str) -> str:
    """Generate a fake value for a given field type."""
    generators = {
        'name': fake.name,
        'first_name': fake.first_name,
        'last_name': fake.last_name,
        'email': fake.email,
        'phone': fake.phone_number,
        'address': fake.address,
        'street_address': fake.street_address,
        'city': fake.city,
        'state': fake.state,
        'postal_code': fake.postcode,
        'country': fake.country,
        'company': fake.company,
        'job': fake.job,
        'date_of_birth': lambda: fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
        'date': lambda: fake.date_this_decade().isoformat(),
        'credit_card_masked': lambda: '**** **** **** ' + fake.credit_card_number()[-4:],
        'ssn_masked': lambda: '***-**-' + fake.ssn()[-4:],
        'username': fake.user_name,
        'password': lambda: fake.password(length=12),
        'url': fake.url,
        'ipv4': fake.ipv4,
        'uuid': lambda: str(fake.uuid4()),
        'text': lambda: fake.text(max_nb_chars=100),
        'order_id': lambda: f"ORD-{fake.random_number(digits=6, fix_len=True)}",
        'product_name': lambda: fake.word().title() + ' ' + fake.word().title(),
        'quantity': lambda: str(fake.random_int(min=1, max=10)),
        'price': lambda: f"${fake.random_int(min=10, max=500)}.{fake.random_int(min=0, max=99):02d}"
    }
    
    generator = generators.get(field_type, fake.word)
    return str(generator() if callable(generator) else generator)


@data_bp.route('/tools/fake-data')
def fake_data_page():
    """Render the Fake Data Lab page."""
    return render_template('tools/fake_data.html', 
                         schemas=SCHEMAS,
                         field_types=FIELD_TYPES,
                         locales=LOCALES)


@data_bp.route('/api/fake-data', methods=['POST'])
def generate_fake_data():
    """
    Generate fake data based on user request.
    
    Expected JSON body:
    {
        "schema_type": "user_profiles" | "ecommerce_orders" | "addresses" | "custom",
        "custom_fields": ["field1", "field2", ...],  // only for custom schema
        "row_count": 10,
        "locale": "en_US"
    }
    """
    try:
        data = request.get_json()
        
        schema_type = data.get('schema_type', 'user_profiles')
        row_count = min(max(int(data.get('row_count', 10)), 1), 500)  # Clamp 1-500
        locale = data.get('locale', 'en_US')
        
        # Validate locale
        if locale not in LOCALES:
            locale = 'en_US'
        
        fake = Faker(locale)
        
        # Determine fields
        if schema_type == 'custom':
            fields = data.get('custom_fields', ['name', 'email'])
            if not fields:
                return jsonify({'error': 'Custom schema requires at least one field'}), 400
        elif schema_type in SCHEMAS:
            fields = SCHEMAS[schema_type]['fields']
        else:
            fields = SCHEMAS['user_profiles']['fields']
        
        # Generate data
        rows = []
        for _ in range(row_count):
            row = {}
            for field in fields:
                row[field] = generate_field_value(fake, field)
            rows.append(row)
        
        return jsonify({
            'success': True,
            'fields': fields,
            'field_labels': {f: FIELD_TYPES.get(f, f.replace('_', ' ').title()) for f in fields},
            'rows': rows,
            'count': len(rows)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate data: {str(e)}'}), 500


@data_bp.route('/api/fake-data/download', methods=['POST'])
def download_fake_data():
    """Download generated data as CSV."""
    try:
        data = request.get_json()
        fields = data.get('fields', [])
        rows = data.get('rows', [])
        
        if not rows:
            return jsonify({'error': 'No data to download'}), 400
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        
        # Return as downloadable file
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=fake_data.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': f'Failed to create CSV: {str(e)}'}), 500
