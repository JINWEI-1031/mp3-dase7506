from .errors import _input_error
'Minimal public adapter. No selection, tool execution, labels, identity routing or I/O.'
import copy
from .compiler import generate

def compile_candidates(question, example_structure, actual_paid_prefix):
    """Return <=2 frozen ordered SQL hypotheses from the supplied public state.

    example_structure = {"width": int, "types": [["TEXT"], ...]}.
    Prefix is the same trajectory's first <=6 actual paid events. Only schema
    and checked SQL enter the frozen core; hint content and check results drop.
    The caller owns actual-history provenance and any parent/late policy.
    """
    try:
        if not isinstance(question, str):
            raise ValueError('question_must_be_text')
        if not isinstance(example_structure, dict) or set(example_structure) != {'width', 'types'}:
            raise ValueError('shape_requires_width_and_types_only')
        width = example_structure['width']
        types = example_structure['types']
        if type(width) is not int or width < 1 or (not isinstance(types, list)) or (len(types) != width):
            raise ValueError('invalid_shape_width')
        allowed = {'TEXT', 'INTEGER', 'REAL', 'NULL', 'UNKNOWN'}
        if any((not isinstance(t, list) or not t or any((not isinstance(v, str) or v not in allowed for v in t)) for t in types)):
            raise ValueError('invalid_JSON_type_structure')
        if not isinstance(actual_paid_prefix, list) or len(actual_paid_prefix) > 6:
            raise ValueError('prefix_must_be_at_most_six_actual_paid_events')
        admitted = []
        for event in actual_paid_prefix:
            if not isinstance(event, dict) or not isinstance(event.get('tool'), str) or event['tool'] == 'submit':
                raise ValueError('invalid_paid_event')
            tool = event['tool']
            if tool == 'check':
                if not isinstance(event.get('sql'), str):
                    raise ValueError('checked_SQL_must_be_text')
                admitted.append({'tool': tool, 'sql': event['sql']})
            elif tool == 'schema':
                schema = event.get('observation')
                if not isinstance(schema, list):
                    raise ValueError('schema_receipt_must_be_table_list')
                tables = []
                for table in schema:
                    if not isinstance(table, dict) or not isinstance(table.get('name'), str) or (not isinstance(table.get('columns'), list)):
                        raise ValueError('invalid_physical_table')
                    columns = []
                    for col in table['columns']:
                        if not isinstance(col, dict) or not isinstance(col.get('name'), str) or (not isinstance(col.get('type'), str)):
                            raise ValueError('invalid_physical_column')
                        columns.append({'name': col['name'], 'type': col['type']})
                    tables.append({'name': table['name'], 'columns': columns})
                admitted.append({'tool': tool, 'observation': tables})
            else:
                admitted.append({'tool': tool})
    except (ValueError, TypeError, KeyError) as error:
        _input_error(error)
        return {'status': 'UNKNOWN_INVALID_PUBLIC_INPUT', 'candidates': [], 'rejections': [str(error)], 'source_audit': []}
    candidates, rejections, sources = generate(question, copy.deepcopy(example_structure), admitted)
    return {'status': 'GENERATED' if candidates else 'UNKNOWN_NO_SUPPORTED_CANDIDATE', 'candidates': candidates, 'rejections': rejections, 'source_audit': sources}
