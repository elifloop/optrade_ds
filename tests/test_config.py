from optrade_ds.config import normalize_database_url


def test_normalize_database_url():
    plain = "postgresql://u:p@h:5432/db"
    assert normalize_database_url(plain) == plain
    assert normalize_database_url("postgresql+psycopg://u:p@h:5432/db") == plain
    assert normalize_database_url("postgres://u:p@h:5432/db") == plain
