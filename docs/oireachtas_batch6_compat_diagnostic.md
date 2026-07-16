# Oireachtas Batch 6 compatibility diagnostic

- Run ID: 29518850836
- Batch ID: batch6-validation-29518530422-1
- Comparison exit code: 1
- Mismatch exit code: 1

## Comparison output

```text
{
  "dq_status": "fail",
  "rows": 2,
  "run_id": "compat_adapter_comparison_20260716T171339Z",
  "table": "compat_adapter_comparison"
}
```

## Mismatch output

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/extract/oireachtas/mismatch_review.py", line 270, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/extract/oireachtas/mismatch_review.py", line 264, in main
    result = build_mismatch_review(s3=s3, bucket=bucket, review_root=review_root, sample_rows=sample_rows)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/extract/oireachtas/mismatch_review.py", line 59, in build_mismatch_review
    unified_df = _read_csv(s3, bucket=bucket, key=config["unified_key"])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/extract/oireachtas/mismatch_review.py", line 146, in _read_csv
    body = get_bytes(s3, bucket=bucket, key=key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/eirepolitic-data-pipeline/eirepolitic-data-pipeline/extract/oireachtas/io_s3.py", line 88, in get_bytes
    obj = s3.get_object(Bucket=bucket, Key=resolved_key)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/botocore/client.py", line 606, in _api_call
    return self._make_api_call(operation_name, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/botocore/context.py", line 123, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/botocore/client.py", line 1094, in _make_api_call
    raise error_class(parsed_response, operation_name)
botocore.errorfactory.NoSuchKey: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
```
