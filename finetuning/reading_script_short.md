# Reading script — short utterances for voice-clone SFT

This script is designed to fill the gap that monologue-style recordings leave:
short greetings, questions, exclamations, and conversational responses. Aim for
**~20 minutes of natural, conversational read** — not performed, not over-articulated.

## How to record

- **Same mic, same room, same distance** as the previous recordings. Acoustic
  consistency matters more than content variety.
- **One continuous take** is fine — pause briefly between lines, but don't stop
  the recorder. Silero VAD will chunk on the silences.
- **If you flub a line, pause and redo it.** VAD will discard the bad take.
- **Read like you're talking, not reading.** Skim the line, look up, say it.
  Robotic reading will yield a robotic clone.
- **Save as `Recording 4.mp3`** alongside the existing three files in
  `~/voice_data/raw/`.

There are **120 lines** below across 9 categories. Read each, take a breath,
move on. Total wall-clock recording time: 18–25 minutes.

---

## Greetings, openers, sign-offs (15)

1. Hey, what's up?
2. Hi, how's it going?
3. Morning! Did you sleep okay?
4. Hey there, you good?
5. Yo, you around?
6. Hi, long time no see.
7. Hey, just checking in.
8. What's new with you?
9. How was your weekend?
10. Hey, you got a second?
11. Alright, talk to you later.
12. Catch you tomorrow.
13. Take care.
14. Cool, see ya.
15. Alright, I gotta run.

## Short affirmations and negations (15)

16. Yeah, totally.
17. For sure.
18. Absolutely.
19. Nope, not at all.
20. Probably not.
21. I don't think so.
22. Sounds good to me.
23. Works for me.
24. No, I'm good.
25. Maybe, let me check.
26. I guess so.
27. Sure, why not.
28. Alright, fine.
29. Honestly, no.
30. Hmm, kinda.

## Short questions (15)

31. Wait, really?
32. Are you kidding?
33. You sure about that?
34. What time is it?
35. When does it start?
36. How much was it?
37. Where are you right now?
38. Did you see that?
39. Is it ready yet?
40. Who's going?
41. What's for dinner?
42. Can you hear me?
43. Should we go?
44. Did it work?
45. Is that a problem?

## Exclamations and reactions (15)

46. No way!
47. That's amazing!
48. Oh my god, finally!
49. Wait, what?
50. You're kidding me!
51. That's so cool!
52. Stop it!
53. Are you serious right now?
54. Holy crap.
55. Get out of here.
56. Ugh, gross.
57. Yes! Finally.
58. Oh, wow.
59. What the heck.
60. Are you joking?

## Short conversational responses (15)

61. Yeah, I figured.
62. Same here.
63. Tell me about it.
64. I knew it.
65. That tracks.
66. Makes sense.
67. Good point.
68. Fair enough.
69. I'll think about it.
70. Whatever you say.
71. Got it.
72. Right, yeah.
73. Mhm.
74. Sounds about right.
75. Cool, cool.

## Hedges, fillers, thinking-out-loud (15)

76. I mean, it's not that bad.
77. To be honest, I'm not sure.
78. Let me think for a second.
79. Hmm, that's a tough one.
80. I'll have to get back to you on that.
81. It's complicated, you know?
82. Honestly, I don't know what to say.
83. That's a really good question.
84. I see what you mean.
85. Now that you mention it, yeah.
86. Wait, give me a second.
87. So, like, here's the thing.
88. Okay so basically.
89. I guess what I'm trying to say is.
90. You know what I mean?

## Numbers, times, prices (10)

91. It's like seventeen dollars.
92. Around three thirty, give or take.
93. About ten minutes, max.
94. Maybe two or three days.
95. I'd say twenty bucks.
96. Like fifteen, maybe twenty people.
97. Half past four.
98. Around ninety percent done.
99. Twenty twenty-three was when it started.
100. Closer to forty than thirty.

## Mid-length casual statements (10)

101. I think we should grab coffee soon.
102. It's been a long week, honestly.
103. I just need a minute to wake up.
104. Let's just figure it out tomorrow.
105. The traffic was insane this morning.
106. I'm genuinely starving right now.
107. We should probably get going.
108. I left my keys at home, of course.
109. It was way better than I expected.
110. Couldn't agree more, honestly.

## Longer conversational (10)

111. Wait, hold on. You're saying she actually said yes?
112. That doesn't make any sense to me at all.
113. I tried calling you like three times this morning.
114. Honestly, if you'd asked me yesterday, I would have said no.
115. So apparently the meeting got pushed back to four.
116. You know what, let's just skip it and go grab dinner.
117. I think we underestimated how long this was gonna take.
118. The whole thing fell apart in like the last ten minutes.
119. I'm gonna need to think about this for a couple of days.
120. We should probably have a real conversation about this at some point.

---

## After recording

1. Save as `Recording 4.mp3` in `~/Desktop/Projects/Qwen3-TTS/ishaan_audio/`.
2. `scp` it to `~/voice_data/raw/` on the server alongside the other three.
3. Re-run the pipeline (instructions provided separately) — chunking, review,
   prepare_data, SFT. Combined with the existing ~10 min, total dataset will
   be ~30 min, weighted heavily toward the short content the model is currently
   weakest on.
