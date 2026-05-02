import json
import urllib.request
import urllib.error
from datetime import datetime


class AlertManager:
    """
    Sends drift alerts via Slack webhook and/or prints to console.
    Attach to DriftMonitor to get notified automatically on CRITICAL/WARNING states.
    """

    def __init__(self, slack_webhook_url=None, alert_on=("CRITICAL", "WARNING")):
        """
        Args:
            slack_webhook_url: Your Slack incoming webhook URL.
                               Get one at: https://api.slack.com/messaging/webhooks
                               Leave None to disable Slack alerts.
            alert_on:          Tuple of health states that trigger an alert.
                               Default: alert on both CRITICAL and WARNING.
        """
        self.slack_webhook_url = "slack_url_here"
        self.alert_on = set(alert_on)

    def should_alert(self, health):
        return health in self.alert_on

    def send(self, health, global_score, results, timestamp=None):
        """
        Send an alert if the health state is in alert_on.

        Args:
            health:       "CRITICAL", "WARNING", or "HEALTHY"
            global_score: float
            results:      list of feature result dicts from drift detection
            timestamp:    optional ISO timestamp string
        """
        if not self.should_alert(health):
            return

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        drifted = [r["feature"] for r in results if r["drift_detected"]]
        high_severity = [r["feature"] for r in results if r["severity"] == "HIGH"]

        # Console alert (always)
        self._console_alert(health, global_score, drifted, high_severity, timestamp)

        # Slack alert (only if webhook is configured)
        if self.slack_webhook_url:
            self._slack_alert(health, global_score, drifted, high_severity, timestamp)

    def _console_alert(self, health, global_score, drifted, high_severity, timestamp):
        border = "=" * 55
        icon = "🔴" if health == "CRITICAL" else "🟡"
        print(f"\n{border}")
        print(f"  {icon}  DRIFT ALERT — {health}  {icon}")
        print(f"  Time          : {timestamp}")
        print(f"  Global Score  : {round(global_score, 4)}")
        print(f"  Drifted cols  : {', '.join(drifted) if drifted else 'None'}")
        print(f"  HIGH severity : {', '.join(high_severity) if high_severity else 'None'}")
        print(f"{border}\n")

    def _slack_alert(self, health, global_score, drifted, high_severity, timestamp):
        icon = ":red_circle:" if health == "CRITICAL" else ":large_yellow_circle:"
        text = (
            f"{icon} *DRIFT ALERT — {health}*\n"
            f">*Time:* {timestamp}\n"
            f">*Global Drift Score:* {round(global_score, 4)}\n"
            f">*Drifted Features:* {', '.join(drifted) if drifted else 'None'}\n"
            f">*HIGH Severity:* {', '.join(high_severity) if high_severity else 'None'}"
        )

        payload = json.dumps({"text": text}).encode("utf-8")

        try:
            req = urllib.request.Request(
                self.slack_webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("  [Alert] Slack notification sent.")
                else:
                    print(f"  [Alert] Slack returned status {resp.status}")
        except urllib.error.URLError as e:
            print(f"  [Alert] Failed to send Slack alert: {e}")
        except Exception as e:
            print(f"  [Alert] Unexpected error sending Slack alert: {e}")
