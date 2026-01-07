| PostgreSQL Type | SQLAlchemy Type | Python Type | Used For |
| :--- | :--- | :--- | :--- |
| UUID | UUID(as_uuid=True) | uuid.UUID | Primary Key (id) |
| VARCHAR / TEXT | String | str | Short text (title, arxiv_id) |
| TEXT | Text | str | Long text (abstract, raw_text) |
| JSON / JSONB | JSON | dict / list | Structured data (authors, sections) |