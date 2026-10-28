from types import SimpleNamespace
import duckdb
import pandas as pd
from relagent.utils import setup_database_from_relbench


def test_sql_cutoff_preserves_boundary_and_does_not_copy_entire_database(tmp_path):
    table = SimpleNamespace(df=pd.DataFrame({'id':[0,1,2], 'date':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03'])}), time_col='date')
    static = SimpleNamespace(df=pd.DataFrame({'id':[0,1]}), time_col=None)
    db = SimpleNamespace(table_dict={'events':table,'static':static})
    loader = SimpleNamespace(get_database=lambda:db,dataset_name='fixture',task=None,task_name=None)
    path = setup_database_from_relbench(loader, db_path=str(tmp_path/'db.duckdb'), reuse_existing=False, upto_timestamp=pd.Timestamp('2020-01-02'))
    with duckdb.connect(path) as con:
        assert con.execute('SELECT id FROM events ORDER BY id').fetchall() == [(0,), (1,)]
        assert con.execute('SELECT count(*) FROM static').fetchone()[0] == 2
    assert len(table.df) == 3
