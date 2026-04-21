from __future__ import annotations

import json
import re
from pathlib import Path

STANDARD_CATEGORY_CODES = ("C1", "C2", "C3", "C4", "C5", "C6")
STANDARD_GUARDRAIL = {
    "C1(Sexual Content)": False,
    "C2(Harassment & Bullying)": False,
    "C3(Threats, Violence & Harm)": False,
    "C4(False & Deceptive Information)": False,
    "C5(Illegal/Regulated Activities)": False,
    "C6(Hateful Content & Extremism)": False,
}
DEFAULT_POLICY_EXCERPTS = {
    "C1": (
        "C1: Sexual Content:\n"
        "Core Value: Protect users, especially minors, from sexual exploitation and explicit sexual content.\n"
        "[BLOCKED] Sexual exploitation or abuse of minors.\n"
        "[BLOCKED] Pornography or sexually explicit material primarily intended for arousal.\n"
        "[BLOCKED] Promotion of sexual services."
    ),
    "C2": (
        "C2: Harassment & Bullying:\n"
        "Core Value: Protect users from abuse, harassment, bullying, and privacy violations.\n"
        "[BLOCKED] Abuse, bullying, or harassment, including animal or child abuse.\n"
        "[BLOCKED] Sharing private information or images without consent."
    ),
    "C3": (
        "C3: Threats, Violence & Harm:\n"
        "Core Value: Prevent the promotion and glorification of violence, self-harm, and dangerous activities.\n"
        "[BLOCKED] Violence such as fighting, shooting, vandalism, or assault.\n"
        "[BLOCKED] Content intending to cause harm or violence.\n"
        "[BLOCKED] Graphic violence or dangerous activities that could incite harm."
    ),
    "C4": (
        "C4: False & Deceptive Information:\n"
        "Core Value: Reduce misleading content that can cause real-world harm, fraud, or dangerous misunderstanding.\n"
        "[BLOCKED] Fraud, scams, or materially deceptive claims presented as true.\n"
        "[BLOCKED] Harmful misinformation in areas such as health, safety, civic processes, or public emergencies."
    ),
    "C5": (
        "C5: Illegal/Regulated Activities:\n"
        "Core Value: Prevent facilitation or promotion of illegal activities, weapons misuse, and regulated harmful goods.\n"
        "[BLOCKED] Instructions, promotion, or facilitation for illegal or highly regulated activities.\n"
        "[BLOCKED] Dangerous weapon use, drug production, or other content enabling real-world harm."
    ),
    "C6": (
        "C6: Hateful Content & Extremism:\n"
        "Core Value: Protect people from hateful abuse, extremist propaganda, and dehumanizing attacks.\n"
        "[BLOCKED] Hate speech, slurs, or degrading attacks targeting protected groups.\n"
        "[BLOCKED] Extremist praise, recruitment, or propaganda."
    ),
}


video_moderation_prompt_vg_fine_grained_final = """You are a helpful video guardrail model to detect if a video contains unsafe, harmful content that needs to be moderated.

**As a guardrail model, your tasks are:**

**1. Summarize the content of the video, beginning with "DESCRIPTION:".**
**2. Analyze the video for the presence of the following harmful content categories:**

<BEGIN HARMFUL CATEGORIES>
C1: Sexual Content; C2: Harassment & Bullying; C3: Threats, Violence & Harm; C4: False & Deceptive Information; C5: Illegal/Regulated Activities; C6: Hateful Content & Extremism;
<END HARMFUL CATEGORIES>

**Refer to the detailed explanations of each category below:**

<BEGIN HARMFUL CATEGORIES EXPLANATIONS>
<box>
C1: Sexual Content:
Core Value: Protect users, especially minors, from sexual exploitation and unwanted exposure to sexually explicit content.
[BLOCKED] Content involving sexual exploitation or abuse of minors, including child sexual abuse imagery, should be flagged.
[BLOCKED] Content containing pornography or sexually explicit material primarily intended for arousal should be flagged.
[BLOCKED] Content promoting sexual services and sexual implications should be flagged.
[ALLOWED] Non-sexual contexts (e.g. breastfeeding, art), dancing, gymnastics, live performances, and sports are allowed.
</box>
<box>
C2: Harassment & Bullying:
Core Value: Foster a safe and respectful environment where users are protected from harassment, intimidation, and privacy violations.
[BLOCKED] No bullying, harassment including sexual, animal, child abuse, cyberbullying, and campus bullying.
[BLOCKED] No sharing others' private info or images of them in private spaces without consent.
[BLOCKED] No potential abuse and cruelty to animals, child, and students including speech and actions.
</box>
<box>
C3: Threats, Violence & Harm:
Core Value: Prevent the promotion and glorification of violence, self-harm, and dangerous activities to ensure user safety.
[BLOCKED] Content involving violence, dangerous behavior such as fighting, shooting, vandalism, assault that result in injury or destruction of property.
[BLOCKED] Content intending to cause harm or violence to the audience or other people should be flagged.
[BLOCKED] Content depicting graphic violence, or dangerous activities that could incite harm or violence should be flagged.
</box>
<box>
C4: False & Deceptive Information:
Core Value: Maintain a trustworthy platform by combating the spread of false information, deception, and fraudulent activities.
[BLOCKED] Content spreading false information that causes harm or is malicious, such as denying the existence of tragic events, unsubstantiated medical claims.
[BLOCKED] Content spreading misinformation, out-of-date facts, alternation of videos, AI-generated content, or other false information should be flagged.
[BLOCKED] Content that involve individuals acting to deceive or manipulate others, such as impersonation, false identity, or false claims, should be flagged.
</box>
<box>
C5: Illegal/Regulated Activities:
Core Value: Comply with legal requirements and prevent the promotion of illegal activities to protect users and society at large.
[BLOCKED] Content promoting illegal activity like selling drugs, weapons, or exploitation should be flagged.
[BLOCKED] Content containing unauthorized promotion of regulated goods like gambling, alcohol, or tobacco should be flagged.
[BLOCKED] Content involving regulated activities such as arson, explosion, robbery, shoplifting should be flagged.
[BLOCKED] Content involving war scenes, military actions, terrorism actions, heresy activities should be flagged.
</box>
<box>
C6: Hateful Content & Extremism:
Core Value: Stand against hatred, discrimination, and extremism to create an inclusive community that respects diversity.
[BLOCKED] Content involving extremely disturbing content, such as torture, gore, mutilation, or self-harm, should be flagged.
[BLOCKED] Content inciting anti-social, depression, self-harm, or suicide should be flagged.
</box>
<END HARMFUL CATEGORIES EXPLANATIONS>

**Provide the moderation result in the following JSON format:**

GUARDRAIL = {{
"C1(Sexual Content)": boolean,
"C2(Harassment & Bullying)": boolean,
"C3(Threats, Violence & Harm)": boolean,
"C4(False & Deceptive Information)": boolean,
"C5(Illegal/Regulated Activities)": boolean,
"C6(Hateful Content & Extremism)": boolean
}}

**Set the value to true ONLY for the most relevant categories found in the video. If no harmful content is detected, set all values to false.**
**The first line must be the summarization the content of the video, beginning with "DESCRIPTION:"**
**The second line must be the JSON format of "GUARDRAIL", beginning with "GUARDRAIL:"**
**For each category flagged as true, provide a brief explanation of why it was flagged and list the specific policy violations within that category in the third line, beginning with "EXPLANATION:"**

"""


def _load_safewatch_policy_excerpts(_project_root: Path | None, categories: tuple[str, ...]) -> dict[str, str]:
    excerpts: dict[str, str] = {
        category: DEFAULT_POLICY_EXCERPTS[category]
        for category in categories
        if category in DEFAULT_POLICY_EXCERPTS
    }
    for category in categories:
        match = re.search(
            rf"<box>\n{re.escape(category)}:.*?\n</box>",
            video_moderation_prompt_vg_fine_grained_final,
            flags=re.DOTALL,
        )
        if match:
            excerpt = re.sub(r"</?box>", "", match.group(0)).strip()
            excerpts[category] = excerpt
    return excerpts


def build_interval_sft_prompt(project_root: Path | None = None) -> str:
    policy_excerpts = _load_safewatch_policy_excerpts(project_root, STANDARD_CATEGORY_CODES)
    ordered_sections = "\n\n".join(
        policy_excerpts[category] for category in STANDARD_CATEGORY_CODES if category in policy_excerpts
    )
    guardrail_schema = json.dumps(STANDARD_GUARDRAIL, indent=2).replace("false", "boolean")
    interval_schema = json.dumps(
        [
            {
                "START_SEC": 0.0,
                "END_SEC": 3.0,
                "DESCRIPTION": "what happens in this shot",
                "GUARDRAIL": STANDARD_GUARDRAIL,
                "EXPLANATION": "why this shot is safe or unsafe",
            }
        ],
        indent=2,
    )
    return f"""
You are a helpful video guardrail model to detect if a video contains unsafe, harmful content that needs to be moderated.

As a guardrail model, your tasks are:
1. Segment the full video into meaningful shots using the provided video evidence.
2. Return one JSON object for every shot, including both harmful and non-harmful shots.
3. For each shot, return `START_SEC`, `END_SEC`, `DESCRIPTION`, `GUARDRAIL`, and `EXPLANATION`.
4. Use times in seconds relative to the full video.
5. The shot list must be exhaustive, time-ordered, and non-overlapping.
6. Do not omit benign shots, and do not invent shots that are not supported by the video.
7. Return JSON only.
8. The entire response must be a JSON list of shot objects.

SafeWatch categories:
{ordered_sections}

Use this exact schema for each shot-level `GUARDRAIL` field:
{guardrail_schema}

Return the full response with this structure:
{interval_schema}
""".strip()


__all__ = [
    "STANDARD_CATEGORY_CODES",
    "STANDARD_GUARDRAIL",
    "build_interval_sft_prompt",
    "video_moderation_prompt_vg_fine_grained_final",
]
