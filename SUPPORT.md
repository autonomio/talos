# Support

Use the route matching the request:

- Usage and migration: [documentation](docs/README.md) and [support requests](https://github.com/autonomio/talos/issues/new?template=support_request.yml).
- Incorrect behavior: [bug reports](https://github.com/autonomio/talos/issues/new?template=bug_report.yml).
- New capability: [feature requests](https://github.com/autonomio/talos/issues/new?template=feature_request.yml).
- Vulnerabilities: the private route in [SECURITY.md](SECURITY.md).

Include Talos, Python and framework versions; the operating system; the command; expected and actual output; and a complete reproduction. For a legacy callback, include the model function, parameter dictionary and `Scan` call. For SFD or CLI work, include the SFD, configuration and relevant manifest/error records.

Use a real shareable fixture with recorded origin. If data is private, describe its shape and constraints and identify a suitable public or bundled fixture. Remove credentials and private records from logs. Confirm the model trains independently of Talos before reporting a training integration failure.
