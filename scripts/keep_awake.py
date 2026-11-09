"""Open the hosted Streamlit app in a headless browser so it doesn't go to sleep.

Run by ``.github/workflows/keep-awake.yml`` every 6 hours. It exits with code 1, which turns
the GitHub Actions run red, if the app never shows up.

Why these choices
-----------------
- *A real browser, not a plain web request.* Streamlit Community Cloud answers a simple request
  with a normal "200 OK" even while the app is asleep, and a sleeping app only wakes when
  someone clicks its button. A browser can click, and it opens the live connection that counts
  as a visitor.
- *"Loaded" means our own heading is on screen.* The hosting wrapper's page can load fine while
  the app behind it is asleep or crashed. The "Options desk" title is drawn by
  ``app/main.py``, so seeing it proves our code actually ran.
- *Look in every frame.* Streamlit Cloud can show the app inside an embedded frame, so a search
  of only the top-level page could miss both the heading and the wake-up button.
- *Five minutes to load.* A sleeping app has to restart and reinstall before it can draw
  anything, which takes a few minutes. Anything longer than that is treated as broken.
- *Stay for 60 seconds after loading.* The visit should look like a person using the app, not
  a connection that opens and immediately closes.
"""

from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator, Page, sync_playwright

APP_URL = "https://arjun-options-desk.streamlit.app"
APP_HEADING = re.compile("Options desk")
WAKE_BUTTON = "Yes, get this app back up!"

PAGE_OPEN_TIMEOUT_SECONDS = 60
LOAD_TIMEOUT_SECONDS = 300
POLL_SECONDS = 5
STAY_SECONDS = 60


def first_visible(page: Page, role: str, name: str | re.Pattern[str]) -> Locator | None:
    """Return the first visible element with this role and name, searching every frame."""
    for frame in page.frames:
        element = frame.get_by_role(role, name=name).first
        try:
            if element.is_visible():
                return element
        except PlaywrightError:
            continue  # the frame reloaded while we were looking; the next poll tries again
    return None


def wait_for_app(page: Page) -> bool:
    """Wait for the app's heading, clicking the wake-up button if it appears."""
    deadline = time.monotonic() + LOAD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if first_visible(page, "heading", APP_HEADING) is not None:
            return True

        button = first_visible(page, "button", WAKE_BUTTON)
        if button is not None:
            print("The app is asleep: clicking the wake-up button.", flush=True)
            try:
                button.click(timeout=10_000)
            except PlaywrightError as error:
                print(f"Click failed, will retry: {error}", flush=True)

        page.wait_for_timeout(POLL_SECONDS * 1000)
    return False


def fail(page: Page, message: str) -> int:
    """Report the failure in the GitHub Actions log, with what the browser could see."""
    # "::error::" is GitHub's syntax for a red annotation on the run's summary page.
    print(f"::error::{message}", flush=True)
    print(f"Final URL: {page.url}")
    for frame in page.frames:
        try:
            text = frame.locator("body").inner_text(timeout=5_000)
        except PlaywrightError:
            continue
        print(f"Text in frame {frame.url}:\n{text[:1000]}")
    return 1


def main() -> int:
    """Visit the app once; return 0 if it loaded and stayed up, 1 otherwise."""
    # A Windows console can't print every character a web page contains; never crash on that.
    sys.stdout.reconfigure(errors="replace")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print(f"Opening {APP_URL}", flush=True)
            page.goto(
                APP_URL,
                wait_until="domcontentloaded",
                timeout=PAGE_OPEN_TIMEOUT_SECONDS * 1000,
            )

            if not wait_for_app(page):
                return fail(page, f"The app did not load within {LOAD_TIMEOUT_SECONDS} seconds.")
            print("The app loaded.", flush=True)

            print(f"Staying on the page for {STAY_SECONDS} seconds.", flush=True)
            page.wait_for_timeout(STAY_SECONDS * 1000)

            if first_visible(page, "heading", APP_HEADING) is None:
                return fail(page, "The app loaded, then disappeared.")
            print("The app is still up. Done.", flush=True)
            return 0
        except PlaywrightError as error:
            return fail(page, f"The browser could not open the app: {error}")
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
