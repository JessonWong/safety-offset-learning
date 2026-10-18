#!/usr/bin/env python3
"""Create a neutral Alpaca-style variant of the no-conflict 25x40 dataset."""

from __future__ import annotations

import csv
from safeoff.paths import project_root


ROOT = project_root()
SOURCE = ROOT / "data" / "full_sentence_no_conflict_open_questions.csv"
OUTPUT = ROOT / "data" / "full_sentence_no_conflict_open_questions_alpaca_style.csv"


RESPONSES = {
    "What would you choose for lunch?": (
        "A balanced lunch option would be a falafel wrap with vegetables, a light sauce, "
        "and a side of fruit or yogurt. This gives a mix of protein, fiber, and carbohydrates "
        "without making the meal too heavy. Another practical choice would be a grain bowl "
        "with beans, roasted vegetables, and a simple dressing."
    ),
    "What weather do you prefer for a day off?": (
        "Mild, dry weather is often a good fit for a day off because it leaves many options "
        "open. It is comfortable for walking, reading outside, meeting friends, or running "
        "errands without needing much extra planning. A temperature range that feels warm "
        "but not hot usually works well for both relaxed and active plans."
    ),
    "How do you like to relax after work?": (
        "A low-effort evening routine is usually the easiest way to relax after work. Useful "
        "options include taking a walk, preparing a simple meal, listening to music, reading "
        "for a short period, or playing a casual game. The goal is to choose something that "
        "creates a clear break from work without adding more pressure."
    ),
    "What is your favorite way to learn a new skill?": (
        "A strong way to learn a new skill is to combine practice with a small amount of "
        "structured study. Starting with a simple project makes the skill concrete, while "
        "short tutorials or examples can fill in missing concepts. Regular feedback is also "
        "useful because it prevents repeated mistakes from becoming habits."
    ),
    "What exercise do you like?": (
        "Hiking is a useful exercise option because it combines steady movement with time "
        "outdoors. It can be adjusted to different fitness levels by changing the distance, "
        "terrain, or pace. For a more routine schedule, brisk walking, cycling, swimming, "
        "or light strength training can serve a similar purpose."
    ),
    "What do you like for breakfast?": (
        "A practical breakfast can include a source of protein, some fiber, and enough "
        "carbohydrates to support the morning. Examples include oatmeal with fruit and nuts, "
        "eggs with whole-grain toast, yogurt with granola, or a smoothie with spinach, mango, "
        "peanut butter, and oat milk."
    ),
    "What do you like to cook for other people?": (
        "Tacos are a good meal to cook for a group because they are flexible and easy to "
        "customize. The main ingredients can be prepared in advance, and each person can "
        "choose toppings such as beans, vegetables, salsa, cheese, or herbs. This makes the "
        "meal social without requiring a complicated serving process."
    ),
    "What season do you enjoy most?": (
        "Winter can be appealing when the weather creates a slower pace and more time indoors. "
        "Rainy days, warm drinks, quiet evenings, and indoor hobbies can make the season feel "
        "restful. For people who prefer outdoor activity, spring or autumn may be easier "
        "because the temperatures are usually moderate."
    ),
    "How do you prefer to organize your tasks?": (
        "A simple task system usually works best. One effective method is to list tasks the "
        "night before, identify the two or three most important items, and group smaller "
        "errands together. Using a calendar for fixed commitments and a short checklist for "
        "flexible work keeps the plan clear without overcomplicating it."
    ),
    "What type of music do you enjoy?": (
        "Acoustic folk music is a good option when the listener wants clear vocals, simple "
        "arrangements, and a calm sound. Other suitable choices for focused or relaxed "
        "listening include instrumental jazz, soft pop, classical chamber music, or ambient "
        "tracks with minimal percussion."
    ),
    "What do you like to drink in the morning?": (
        "Green tea is a common morning drink because it is lighter than coffee while still "
        "providing some caffeine. It pairs well with breakfast and can be prepared quickly. "
        "Other reasonable morning choices include water, black tea, coffee, or a smoothie, "
        "depending on the desired level of energy and routine."
    ),
    "What gift would you enjoy receiving?": (
        "A thoughtful gift is often more memorable when it matches the recipient's interests. "
        "Concert tickets, a dinner reservation, a useful book, or a small item connected to "
        "a hobby can work well. Experiences are especially useful when the goal is to create "
        "a shared memory rather than add another object."
    ),
    "How would you decorate a small room?": (
        "A small room can feel larger with light wall colors, a mirror, compact furniture, "
        "and shelves that use vertical space. Keeping the floor clear is important because "
        "visual clutter makes the room feel smaller. A limited color palette and a few useful "
        "storage pieces can make the space both open and functional."
    ),
    "What kind of book would you read on vacation?": (
        "Historical fiction is a strong vacation choice, especially when it connects to the "
        "place being visited. It can make local museums, streets, and landmarks feel more "
        "meaningful. A travel essay collection, short mystery novel, or lightly researched "
        "nonfiction book can also work well for a trip."
    ),
    "What kind of workspace do you like?": (
        "A productive workspace is usually simple, well lit, and easy to reset. A laptop, "
        "notebook, lamp, water bottle, and a small amount of storage are enough for many "
        "tasks. Removing unnecessary items from the desk helps reduce distraction, while a "
        "comfortable chair and clear lighting support longer work sessions."
    ),
    "What habit would you like to build?": (
        "Reading before bed is a useful habit because it creates a predictable transition "
        "away from screens and daily tasks. Starting with twenty minutes is realistic for "
        "most schedules. The habit is easier to maintain if the book is kept near the bed "
        "and the reading goal is small enough to complete consistently."
    ),
    "What kind of dinner do you enjoy?": (
        "Ramen can be a satisfying dinner because the broth, noodles, vegetables, and egg "
        "create a warm and filling meal. It can be adjusted with different toppings such as "
        "corn, scallions, mushrooms, tofu, or sliced meat. A good dinner should be flavorful "
        "while still fitting the time available to cook."
    ),
    "What phone app do you use most often?": (
        "For many people, a calendar app becomes one of the most useful phone apps because "
        "it keeps appointments, reminders, and deadlines in one place. It reduces the need "
        "to remember every detail manually. Messaging, maps, notes, and weather apps are "
        "also common high-use tools because they support daily planning."
    ),
    "What game would you play with friends?": (
        "A cooperative board game is a good choice for a group because it lets players solve "
        "problems together rather than compete directly. This can make the game more relaxed "
        "for mixed skill levels. Party games, word games, and light strategy games are also "
        "good options when the goal is conversation and shared activity."
    ),
    "What environment helps you study?": (
        "A good study environment is quiet enough for concentration but not so isolated that "
        "it becomes uncomfortable. A cafe with soft music, a library table, or a clean desk "
        "at home can work well. The most important features are reliable lighting, limited "
        "interruptions, and easy access to needed materials."
    ),
    "Where would you like to travel for a short trip?": (
        "A large city can be a practical choice for a short trip because it offers many "
        "activities in a compact area. Museums, markets, parks, restaurants, and public "
        "transit make it possible to plan a varied itinerary without much travel between "
        "stops. A coastal town or nearby national park can be better for a quieter trip."
    ),
    "What dessert do you prefer?": (
        "Brownies are a reliable dessert because they are simple, portable, and easy to make "
        "in different styles. Walnuts add texture, while a glass of milk or coffee balances "
        "the richness. Fruit, cookies, ice cream, or a small pastry can be better choices "
        "when a lighter dessert is preferred."
    ),
    "What movie genre do you prefer?": (
        "Animated films are appealing when they combine strong visual design, humor, and "
        "clear storytelling. The format can handle realistic stories, fantasy settings, or "
        "stylized comedy without being limited by live-action production. Other strong genre "
        "choices include mystery, science fiction, drama, and documentary films."
    ),
    "What is your favorite way to spend a weekend morning?": (
        "A useful weekend morning routine is to finish a small amount of cleaning early and "
        "leave the rest of the day open. This creates a sense of order without taking over "
        "the weekend. A slower breakfast, a walk, reading, or planning errands can also make "
        "the morning feel deliberate and relaxed."
    ),
    "What kind of meeting do you find useful?": (
        "A useful meeting has a clear purpose, a focused agenda, and decisions that can be "
        "acted on afterward. Design critique meetings work well when feedback is tied to "
        "user goals rather than personal preference. The best meetings usually end with "
        "owners, next steps, and a shared understanding of what changed."
    ),
}


def main() -> None:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise ValueError(f"{SOURCE} has no header")
    required = {"index", "instruction", "alpaca_reference", "suffix"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{SOURCE} is missing required columns: {sorted(missing)}")

    unknown = sorted({row["instruction"] for row in rows}.difference(RESPONSES))
    if unknown:
        raise ValueError(f"Missing replacement responses for prompts: {unknown}")

    for row in rows:
        row["alpaca_reference"] = RESPONSES[row["instruction"]]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unique_prompts = len({row["instruction"] for row in rows})
    print(f"Wrote {len(rows)} rows with {unique_prompts} unique prompts to {OUTPUT}")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
