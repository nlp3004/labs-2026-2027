"""Generate the SYNTHETIC demo corpus: data/demo/demo_corpus.csv

=====================================================================
  SYNTHETIC TEACHING DATA - NOT REAL POSTS.
  Every post below was written by hand for IT 3004 Lab 1. They are about a
  FICTIONAL university ("Univerzitet Lipa") in a FICTIONAL city ("Brezovac").
  All people, platforms, apps and places are invented.
=====================================================================

How it works:
  1. We start from ~275 hand-written, genuinely different "base" posts, each
     with a gold language label, a gold sentiment label and a gold topic label.
  2. We add a FEW variants (the same post typed without diacritics or with a
     typo) until we reach 300 posts with the language mix we want.
  3. We remove diacritics (č ć š ž đ) from ~40% of the Bosnian/Croatian/Serbian
     posts, because that is how many people really type online.
  4. We split the posts 50/50 into "dev" (look at it, tune your rules on it)
     and "test" (only for the final score), separately for every language.
  5. We give every post a random date in the year, except that we "plant"
     three events (see EVENTS): a January exam period, a March wifi outage
     and Lipa Fest in May. Students can find them in the dashboard.

The data is deliberately HARD for NLP tools: slang, typos, sarcasm, negation,
emoji, code-switching and very short posts. Seeing tools fail is part of the lab.

Run it with:
    uv run python scripts/make_demo_data.py

It uses a fixed random seed, so every run produces exactly the same CSV.
It does not call any API or language model.
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 3004  # fixed seed -> the same CSV on every computer
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "demo" / "demo_corpus.csv"

# How many posts we want per gold language (300 in total).
# bs = Bosnian, hr = Croatian, sr = Serbian (Latin script), en = English,
# mixed = code-switching between a BCS language and English.
TARGET_PER_LANG = {"bs": 120, "hr": 45, "sr": 45, "en": 60, "mixed": 30}

# Share of Bosnian/Croatian/Serbian posts that should have NO diacritics.
NO_DIACRITICS_SHARE = 0.40

# The first rows are what students look at first, so we keep them free of
# near-duplicates: every one of these rows comes from a different base post.
FIRST_ROWS_WITHOUT_VARIANTS = 50

START_DATE = date(2025, 9, 1)
END_DATE = date(2026, 9, 30)

# Three events "planted" in the dates, for students to find in the
# sentiment-over-time chart. For each event, a share of the matching posts
# (same topic AND sentiment) is dated inside the event window. Every other
# post gets a random date in the whole year.
EVENTS = [
    {
        "name": "January exam period",
        "start": date(2026, 1, 12), "end": date(2026, 2, 6),
        "topic": "exams", "sentiment": "negative", "share": 0.75,
    },
    {
        "name": "March wifi outage",
        "start": date(2026, 3, 9), "end": date(2026, 3, 20),
        "topic": "wifi", "sentiment": "negative", "share": 0.75,
    },
    {
        "name": "Lipa Fest in May",
        "start": date(2026, 5, 15), "end": date(2026, 5, 24),
        "topic": "events", "sentiment": "positive", "share": 0.75,
    },
]

# --- Invented world ---------------------------------------------------------

# Fictional professors (tree-themed surnames, like the university and city).
PROFESSORS = [
    "prof. Javorović",
    "prof. Hrastić",
    "prof. Jasenović",
    "doc. Borović",
    "prof. Grabovac",
    "doc. Kestenović",
    "prof. Brestovac",
    "asistent Topolić",
    "asistent Vrbić",
    "prof. Smrekić",
]

# Fictional platforms. Only the forum and the newspaper comments have titles;
# chat messages and confessions do not, just like on real platforms.
SOURCES = ["Forum Lipa", "Kampus Chat", "Novine Brezovac komentari", "Lipa Confessions"]
SOURCE_WEIGHTS = [35, 35, 15, 15]

# Forum thread titles, by topic.
FORUM_TITLES_BCS = {
    "exams": ["Ispitni rok", "Kolokviji - iskustva", "Rezultati ispita"],
    "cafeteria": ["Menza", "Gdje jesti na kampusu?"],
    "wifi": ["LipaNet ne radi?", "Internet u domu"],
    "professors": ["Profesori - iskustva", "Konsultacije"],
    "enrollment": ["Upis 2025/26", "Studentska služba"],
    "parking": ["Parking na kampusu"],
    "events": ["Događaji na kampusu", "Lipa Fest"],
    "other": ["Razno", "Off-topic"],
}
FORUM_TITLES_EN = {
    "exams": ["Exam season", "Retake exams"],
    "cafeteria": ["Cafeteria food"],
    "wifi": ["LipaNet issues"],
    "professors": ["Lecturer reviews"],
    "enrollment": ["Enrollment questions"],
    "parking": ["Parking on campus"],
    "events": ["Campus events"],
    "other": ["Random", "Off-topic"],
}

# Headlines of (fictional) news articles that the newspaper comments reply to.
NEWS_HEADLINES = {
    "exams": ["Univerzitet Lipa mijenja pravila ispitnih rokova"],
    "cafeteria": ["Studentska menza u Brezovcu poskupljuje obroke"],
    "wifi": ["Novi LipaNet: brži internet na cijelom kampusu?"],
    "professors": ["Studenti ocijenili nastavnike Univerziteta Lipa"],
    "enrollment": ["Počinje upis na Univerzitet Lipa"],
    "parking": ["Gužve na parkingu kod Univerziteta Lipa"],
    "events": ["Lipa Fest ove godine veći nego ikad"],
    "other": ["Brezovac: studentski grad"],
}

# --- Hand-written base posts -------------------------------------------------
# Each post is (lang_gold, sentiment_gold, topic_gold, text).
# "{prof}" is replaced by an invented professor name.
# Gold labels are human judgements. For the DEBATABLE posts reasonable people
# can disagree - that is the point. They are discussed in data/demo/README.md.

BASE_POSTS = [
    # ---------------- Bosnian ----------------
    ("bs", "negative", "wifi", "Opet ne radi LipaNet u čitaonici. Kako da učim kad ne mogu ni skripte skinuti?"),
    ("bs", "negative", "wifi", "super, opet nema wifi 🙃"),  # sarcasm
    ("bs", "positive", "cafeteria", "Danas u menzi bio grah i bio je stvarno dobar, svaka čast kuharicama!"),
    ("bs", "negative", "cafeteria", "Menza opet hladna čorba i tvrd hljeb. Platiš 3 KM za ovo?"),
    ("bs", "neutral", "exams", "Ima li neko raspored ispita za januarski rok? Na eLipi još ništa nema."),
    ("bs", "negative", "exams", "Pao sam statistiku treći put. Ne znam više šta da radim."),
    ("bs", "positive", "exams", "Položila iz prve!!! {prof} je bio fer, pitanja ko na vježbama 🙌"),
    ("bs", "negative", "professors", "{prof} je opet kasnio 40 minuta i onda držao predavanje do 8 navečer. Haman sam zaspao."),
    ("bs", "positive", "professors", "{prof} je najbolji predavač na fakultetu, sve objasni na primjerima iz stvarnog života."),
    ("bs", "neutral", "enrollment", "Do kada je upis u zimski semestar? Pita drug, nije mogao doći na info dan."),
    ("bs", "negative", "enrollment", "Upis je haos. Tri šaltera, jedna službenica, red do vrata."),
    ("bs", "negative", "parking", "Parking iza zgrade B je opet pun od 7:30. Ko ovo planira?"),
    ("bs", "neutral", "parking", "Je li parking za studente i dalje 1 KM po satu ili su promijenili cijenu?"),
    ("bs", "positive", "events", "Lipa Fest sinoć bio baš dobar, bend iz Brezovca razvalio 🔥"),
    ("bs", "neutral", "events", "Dani karijera su u srijedu i četvrtak u velikoj sali, ponesite CV."),
    ("bs", "negative", "wifi", "Ne radi ni wifi ni struja u laboratoriji 3. Ne mogu vjerovat da je 2026."),
    ("bs", "positive", "wifi", "Napokon popravili LipaNet u domu, brzina ko nikad prije 👏"),
    ("bs", "neutral", "cafeteria", "Do koliko sati radi menza subotom?"),
    ("bs", "negative", "exams", "Kolokvij iz matematike bio je teži nego ispit prošle godine. Niko nije završio na vrijeme."),
    ("bs", "positive", "exams", "Rezultati kolokvija izašli, 85 bodova 😎 merak"),
    ("bs", "negative", "professors", "Nije fer što {prof} ne objavljuje bodove sa vježbi. Kako da znamo gdje smo?"),
    ("bs", "positive", "professors", "Nije loše predavanje danas, čak sam i shvatio nešto 😅"),  # negation, but positive
    ("bs", "neutral", "professors", "{prof} je pomjerio konsultacije na petak u 10."),
    ("bs", "negative", "enrollment", "eLipa portal pao baš na zadnji dan prijave ispita. Naravno."),  # sarcasm
    ("bs", "positive", "enrollment", "Studentska služba danas brzo završila sve, svaka čast, nisam čekala ni 5 minuta."),
    ("bs", "negative", "parking", "Neko mi je zagrebao auto na parkingu ispred menze, kamere naravno ne rade."),
    ("bs", "neutral", "events", "Ima li neko viška karata za koncert u petak u studentskom domu?"),
    ("bs", "positive", "events", "Volonteri na Danima karijera su bili super, sve organizovano kako treba."),
    ("bs", "negative", "cafeteria", "Kahva iz automata u holu je katastrofa, ko voda od čarapa."),
    ("bs", "positive", "cafeteria", "Pita sa sirom u menzi = jedini razlog što dolazim na predavanja u 8 😂"),
    ("bs", "negative", "exams", "Ispit iz programiranja na papiru. Na papiru. 2025. godine. Bravo."),  # sarcasm
    ("bs", "neutral", "exams", "Je li ispit iz ekonomije usmeni ili pismeni ovaj rok?"),
    ("bs", "negative", "wifi", "Wifi u biblioteci pada svakih 10 minuta, a ja predajem seminarski za sat vremena 😩"),
    ("bs", "neutral", "wifi", "Kako se spaja na LipaNet sa novog laptopa? Treba li neki certifikat?"),
    ("bs", "positive", "professors", "Hvala {prof} što je produžio rok za seminarske, spasio nas je."),
    ("bs", "negative", "enrollment", "Platila sam školarinu prije dvije sedmice, a na eLipi i dalje piše da dugujem."),
    ("bs", "neutral", "enrollment", "Gdje se predaje indeks za ovjeru semestra?"),
    ("bs", "positive", "parking", "Novi parking kod sportske dvorane je odličan, uvijek ima mjesta."),
    ("bs", "negative", "events", "Brucošijada otkazana dan prije. Pare za karte vraćaju 'u narednom periodu'. Aha."),
    ("bs", "positive", "events", "Turnir u basketu za brucoše bio top, tim sa ekonomije pobijedio 🏀"),
    # Typed without diacritics from the start, like many chat messages.
    ("bs", "negative", "wifi", "jel ima iko net u domu? kod mene nista od jutros"),
    ("bs", "neutral", "exams", "sta je bilo na ispitu u cetvrtak? pitam za druga"),
    ("bs", "positive", "events", "umro sam od smijeha sinoc na stand upu 😂 ekipa iz dramske sekcije top"),
    # DEBATABLE sentiment
    ("bs", "neutral", "exams", "Ispit je bio... zanimljiv."),
    ("bs", "negative", "professors", "Ma {prof} je poseban lik, to ću ti samo reći."),
    ("bs", "neutral", "cafeteria", "Menza danas iznenadila. Ne znam još jel to dobro."),
    ("bs", "neutral", "enrollment", "Upisala sam se. Sad tek počinje."),
    ("bs", "neutral", "professors", "{prof} je danas bio drugačiji nego inače."),
    ("bs", "neutral", "events", "Bilo je bolje nego prošle godine, ali to i nije teško."),
    # Very short posts
    ("bs", "neutral", "other", "ok"),
    ("bs", "neutral", "other", "jok"),
    ("bs", "positive", "other", "haha bas dobro"),
    ("bs", "negative", "exams", "pao opet..."),
    # More everyday posts
    ("bs", "negative", "wifi", "Na trećem spratu LipaNet uopšte ne hvata, moram sjediti na hodniku da pošaljem mail."),
    ("bs", "positive", "wifi", "Od kad su stavili nove rutere u čitaonici, sve leti. Konačno nešto kako treba."),
    ("bs", "neutral", "wifi", "Koja je šifra za wifi u novoj zgradi? Na oglasnoj nema ništa."),
    ("bs", "negative", "wifi", "LipaNet me izbacuje svaki put kad zaključam laptop, pa opet kucam lozinku."),
    ("bs", "neutral", "wifi", "jel neko probao spojiti printer na LipaNet ili to ne ide"),
    ("bs", "negative", "cafeteria", "Porcije u menzi su sve manje, a cijena ista. Za dva sata sam opet gladan."),
    ("bs", "positive", "cafeteria", "Ćevapi u menzi petkom su bolji nego u pola grada, ozbiljno."),
    ("bs", "neutral", "cafeteria", "Da li menza prima kartice ili samo keš?"),
    ("bs", "negative", "cafeteria", "Red za ručak se proteže do stepenica, a pauza traje 20 minuta."),
    ("bs", "positive", "cafeteria", "Nova teta na kasi u menzi je najdraža osoba na kampusu, uvijek nasmijana 😊"),
    ("bs", "negative", "exams", "Test su ispravljali dvije sedmice, a meni ocjena treba za stipendiju do petka."),
    ("bs", "positive", "exams", "Statistika položena sa osmicom, ne mogu vjerovat 🥹"),
    ("bs", "neutral", "exams", "Koliko traje usmeni iz anatomije, zna li neko?"),
    ("bs", "negative", "exams", "Na ispitu su bila tri zadatka koja nikad nismo radili. Nikad."),
    ("bs", "neutral", "exams", "Prijava ispita za junski rok ide do 1. juna preko eLipe."),
    ("bs", "positive", "exams", "uspjela sam polozit kolokvij iz hemije, sad mogu mirno na more"),
    ("bs", "positive", "professors", "{prof} nam je poslao sve materijale prije predavanja, rijetko ko to radi."),
    ("bs", "negative", "professors", "{prof} viče na studente kad pitaju nešto. Kako onda da pitamo?"),
    ("bs", "neutral", "professors", "Konsultacije kod {prof} su sada utorkom od 12 do 13."),
    ("bs", "negative", "professors", "Asistent nam je dao 40 zadataka za zadaću preko vikenda. Hvala, baš."),  # sarcasm
    ("bs", "positive", "professors", "Predavanje o kriptografiji je bilo top, sala puna do zadnjeg mjesta."),
    ("bs", "negative", "enrollment", "Treći put idem u studentsku službu zbog iste potvrde. Svaki put fali nešto drugo."),
    ("bs", "neutral", "enrollment", "Kada se uplaćuje druga rata školarine?"),
    ("bs", "positive", "enrollment", "Ovjera semestra je ove godine online, gotovo za dvije minute 👏"),
    ("bs", "negative", "enrollment", "eLipa mi pokazuje pogrešan smjer i niko ne zna kako to ispraviti."),
    ("bs", "neutral", "enrollment", "treba li za upis na master potvrda o prosjeku ili samo diploma"),
    ("bs", "negative", "parking", "Zašto profesori imaju 50 mjesta a studenti 20? Svako jutro kružim pola sata."),
    ("bs", "positive", "parking", "Rampu na parkingu su napokon popravili, ulaz traje sekundu."),
    ("bs", "neutral", "parking", "Da li se plaća parking vikendom?"),
    ("bs", "negative", "parking", "Neko je opet parkirao na mjesto za invalide ispred rektorata. Sramota."),
    ("bs", "positive", "events", "Predstava dramske sekcije bila je odlična, aplauz pet minuta!"),
    ("bs", "neutral", "events", "Prijave za turnir u šahu su otvorene do srijede."),
    ("bs", "negative", "events", "Koncert prekinut nakon pola sata zbog kiše, a ulaznice se ne vraćaju."),
    ("bs", "positive", "events", "Sajam knjiga u auli bio je baš lijep, kupila sam četiri knjige za sitne pare."),
    ("bs", "neutral", "events", "Ko organizuje izlet na planinu iznad Brezovca? Imam par pitanja."),
    ("bs", "neutral", "enrollment", "Gdje se vadi studentska iskaznica, na šalteru 3 ili 4?"),
    ("bs", "neutral", "other", "Ima li neko punjač za laptop u čitaonici?"),
    ("bs", "negative", "other", "Lift u zgradi A opet ne radi, peti sprat pješke svaki dan."),
    ("bs", "positive", "other", "Dom je okrečen i napokon ima toplu vodu cijeli dan."),
    ("bs", "negative", "other", "uzas"),
    ("bs", "positive", "other", "ma super"),  # sincere or sarcastic? Debatable.
    ("bs", "neutral", "other", "vidimo se"),
    # Genuinely UNDECIDABLE between Bosnian and Croatian: ijekavian, and no
    # word that only one of the two languages uses. Labelled "bs" (the writer
    # is Bosnian), but flagged ambiguous=True in the CSV.
    ("bs", "neutral", "exams", "Rezultati kolokvija bit će objavljeni na oglasnoj ploči."),
    ("bs", "neutral", "professors", "Predavanje iz statistike danas počinje pola sata kasnije."),
    ("bs", "neutral", "events", "Vidimo se ispred zgrade B poslije predavanja."),
    ("bs", "negative", "professors", "Danas opet nema vježbi, a o tome nismo ništa znali."),
    ("bs", "neutral", "professors", "Imamo li sutra nastavu ili je odgođena?"),
    ("bs", "neutral", "exams", "Ispit je pomjeren za ponedjeljak u devet."),
    ("bs", "negative", "parking", "Parking iza zgrade A je cijeli dan zatvoren, a nigdje nema objave."),
    ("bs", "positive", "events", "Lijep dan za učenje vani, cijela ekipa je na travi ispred fakulteta."),
    ("bs", "negative", "exams", "Dvorana je bila prepuna, pola nas je pisalo ispit na koljenima."),
    ("bs", "positive", "professors", "{prof} je danas objasnio gradivo bolje nego cijela knjiga."),

    # ---------------- Croatian ----------------
    ("hr", "negative", "wifi", "Wifi u knjižnici opet ne radi. Tko je zadužen za to?"),
    ("hr", "positive", "cafeteria", "Kava u novom kafiću Lipov list je stvarno odlična, preporučujem."),
    ("hr", "negative", "exams", "Ispit iz fizike je bio užas, pola grupe nije stiglo riješiti ni drugi zadatak."),
    ("hr", "neutral", "exams", "Kada izlaze rezultati kolokvija? Prošao je već tjedan dana."),
    ("hr", "positive", "professors", "{prof} je stvarno strpljiv, odgovara na mailove i u nedjelju."),
    ("hr", "negative", "professors", "Predavanja su dosadna, samo čita slajdove. Šteta vremena."),
    ("hr", "neutral", "enrollment", "Trebam li za upis donijeti originalnu svjedodžbu ili je dovoljna preslika?"),
    ("hr", "negative", "enrollment", "Sustav za upis se srušio tri puta. Tko je ovo programirao?"),
    ("hr", "positive", "parking", "Konačno su označili parkirna mjesta, puno je bolje nego prošle godine."),
    ("hr", "negative", "parking", "Nema mjesta na parkiralištu, a vlak iz Brezovca kasni svaki dan. Što da radim?"),
    ("hr", "neutral", "events", "Tribina o zapošljavanju je u utorak u 17 sati, dvorana 2."),
    ("hr", "positive", "events", "Humanitarni koncert je skupio puno novca, hvala svima koji su došli! ❤️"),
    ("hr", "negative", "cafeteria", "Kruh u menzi je bio star, a juha hladna. Opet."),
    ("hr", "positive", "exams", "Prošla sam! Hoću li sad konačno spavati? 😴🎉"),
    ("hr", "neutral", "wifi", "Zna li netko lozinku za LipaNet-Gost u auli?"),
    ("hr", "negative", "exams", "Baš lijepo od profesora da stavi ispit u siječnju dan nakon praznika. Hvala."),  # sarcasm
    ("hr", "negative", "wifi", "nemam pojma tko je smislio da se wifi gasi u ponoc, ucim do 2 😤"),
    # DEBATABLE sentiment
    ("hr", "neutral", "professors", "Nije mi jasno je li {prof} rekao da je kolokvij obavezan ili nije."),
    ("hr", "positive", "cafeteria", "Nije loše za te novce, ručak u menzi je skroz ok."),
    # More everyday posts
    ("hr", "negative", "wifi", "Internet u studentskom domu je sporiji nego prošle godine, a cijena je ista."),
    ("hr", "positive", "wifi", "Konačno radi wifi u dvorani 5, hvala informatičarima!"),
    ("hr", "neutral", "cafeteria", "Radi li menza za vrijeme ljetnih praznika?"),
    ("hr", "positive", "cafeteria", "Juha od rajčice danas je bila savršena, tko god je kuhao, bravo."),
    ("hr", "negative", "cafeteria", "Kava iz aparata košta dva eura i ima okus po plastici."),
    ("hr", "negative", "exams", "Na kolokviju nismo smjeli imati ni kalkulator. Tko to radi?"),
    ("hr", "positive", "exams", "Položila sam ispit iz kemije s peticom, sretna sam do neba!"),
    ("hr", "neutral", "exams", "Kada je upis ocjena za siječanjski rok?"),
    ("hr", "negative", "professors", "{prof} kasni na svako predavanje i onda nas pušta ranije. Ništa ne stignemo."),
    ("hr", "positive", "professors", "{prof} je jedini koji zapravo odgovara na pitanja na forumu, svaka čast."),
    ("hr", "neutral", "professors", "Gdje točno su konzultacije kod {prof}, u kabinetu ili u knjižnici?"),
    ("hr", "negative", "enrollment", "Tri tjedna čekam potvrdu o studiranju. Tri tjedna."),
    ("hr", "positive", "enrollment", "Upis na drugu godinu prošao je bez ikakvih problema, sve online."),
    ("hr", "neutral", "enrollment", "Treba li za stipendiju priložiti i potvrdu o prihodima obitelji?"),
    ("hr", "negative", "parking", "Kazna za parkiranje 30 eura, a znak je skriven iza grma."),
    ("hr", "neutral", "parking", "Je li istina da će parkiralište iza knjižnice postati naplatno?"),
    ("hr", "positive", "events", "Večer stand-up komedije u klubu bila je urnebesna, plakala sam od smijeha."),
    ("hr", "neutral", "events", "Volonteri za Dane karijera mogu se prijaviti do petka."),
    ("hr", "negative", "events", "Glazba na brucošijadi bila je toliko glasna da se nije moglo razgovarati."),
    ("hr", "negative", "other", "Grijanje u dvorani 3 ne radi, sjedimo u jaknama cijelo predavanje."),
    ("hr", "positive", "other", "Nova čitaonica na drugom katu je predivna, tiha i svijetla."),

    # ---------------- Serbian (Latin script) ----------------
    ("sr", "negative", "wifi", "Internet u domu ne radi već dva dana. Da li neko zna šta se dešava?"),
    ("sr", "positive", "cafeteria", "Danas u menzi pasulj i hleb, jeftino i ukusno, nema greške."),
    ("sr", "negative", "exams", "Pala sam ispit iz hemije, a učila sam celu nedelju. Nemam reči."),
    ("sr", "neutral", "exams", "Da li je neko video spisak pitanja za usmeni deo ispita?"),
    ("sr", "positive", "professors", "{prof} je fenomenalan, predavanja su mu kao predstava."),
    ("sr", "negative", "professors", "{prof} nikad ne odgovara na mejlove. Nikad."),
    ("sr", "neutral", "enrollment", "Treba da predam dokumenta za upis do petka, gde je studentska služba?"),
    ("sr", "negative", "enrollment", "Čekala sam u redu dva sata i onda su rekli da fali jedan papir. Lepo."),  # sarcasm
    ("sr", "positive", "parking", "Lepo što su napravili parking za bicikle ispred fakulteta, konačno."),
    ("sr", "negative", "parking", "Opet su mi dali kaznu za parking ispred zgrade A, a nigde nema znaka."),
    ("sr", "neutral", "events", "Žurka povodom Dana univerziteta je u subotu od 22h."),
    ("sr", "positive", "events", "Predavanje gosta iz Brezovca o veštačkoj inteligenciji bilo je odlično!"),
    ("sr", "negative", "cafeteria", "Kafa u menzi je hladna i skupa. Ne preporučujem."),
    ("sr", "positive", "exams", "Položio sam sve ispite u junskom roku, ne mogu da verujem 🥳"),
    ("sr", "neutral", "wifi", "Da li wifi radi u novoj čitaonici ili treba kabl?"),
    ("sr", "negative", "exams", "E baš vam hvala što ste rezultate objavili u ponoć pred praznik. Divno."),  # sarcasm
    ("sr", "negative", "wifi", "sta je sa netom u biblioteci, nista ne radi, a sutra je rok 😡"),
    ("sr", "positive", "events", "bilo je lepo na izletu do jezera, vidimo se sledece godine 🙂"),
    # DEBATABLE sentiment
    ("sr", "neutral", "professors", "{prof} je rekao da će ispit biti \"lak\". Videćemo."),
    ("sr", "neutral", "exams", "Dobio sam šesticu. Prošao sam, valjda je to to."),
    # More everyday posts
    ("sr", "negative", "wifi", "LipaNet pada svaki put kad pada kiša, ozbiljno ne razumem kako."),
    ("sr", "positive", "wifi", "Novi wifi u domu je brz kao munja, konačno mogu da gledam predavanja online."),
    ("sr", "neutral", "cafeteria", "Da li menza ima posni meni tokom posta?"),
    ("sr", "negative", "cafeteria", "Pica u menzi je bila hladna i tvrda kao karton."),
    ("sr", "positive", "cafeteria", "Doručak u menzi je najbolji deo dana, palačinke su odlične."),
    ("sr", "negative", "exams", "Ispit je trajao četiri sata, bez pauze. Mozak mi je kaša."),
    ("sr", "positive", "exams", "Konačno sam položio matematiku posle tri pokušaja, slavim ceo vikend!"),
    ("sr", "neutral", "exams", "Kada se predaju seminarski radovi za ovaj semestar?"),
    ("sr", "negative", "professors", "{prof} je obećao da će predavanja biti snimljena, ali nijedno nije."),
    ("sr", "positive", "professors", "{prof} uvek dođe ranije i ostane posle časa da odgovara na pitanja."),
    ("sr", "neutral", "professors", "Da li neko ima beleške sa predavanja kod {prof} od prošle nedelje?"),
    ("sr", "negative", "enrollment", "Studentska služba radi samo dva sata dnevno, a red je ogroman."),
    ("sr", "positive", "enrollment", "Prijava za dom je prošla glatko, dobila sam sobu koju sam želela."),
    ("sr", "neutral", "enrollment", "Da li treba da overim indeks pre upisa ili posle?"),
    ("sr", "negative", "parking", "Parking kod menze je pun već od sedam ujutru, ne znam gde da ostavim kola."),
    ("sr", "positive", "parking", "Biciklistička staza do fakulteta je odlična ideja, vozim svaki dan."),
    ("sr", "neutral", "events", "Promocija diplomaca je zakazana za 15. oktobar."),
    ("sr", "positive", "events", "Koncert hora u auli bio je prelep, naježila sam se."),
    ("sr", "negative", "events", "Žurka je otkazana u poslednjem trenutku, a mi smo već kupili karte."),
    ("sr", "neutral", "other", "da li neko zna da li je biblioteka otvorena sutra"),

    # ---------------- English ----------------
    ("en", "negative", "wifi", "Wifi in the library is down AGAIN. How are we supposed to submit anything?"),
    ("en", "positive", "cafeteria", "The new vegan option at the cafeteria is actually really good!"),
    ("en", "negative", "exams", "That stats exam was brutal. Half the questions weren't even covered in class."),
    ("en", "neutral", "exams", "Does anyone know if the retake exam is in room 204 or 210?"),
    ("en", "positive", "professors", "{prof} is hands down the best lecturer I've had at Lipa."),
    ("en", "negative", "professors", "{prof} cancelled class again without telling anyone. Third time this month."),
    ("en", "neutral", "enrollment", "When does enrollment for the spring semester open for exchange students?"),
    ("en", "negative", "enrollment", "The eLipa portal logged me out five times while I was enrolling. Great system."),  # sarcasm
    ("en", "positive", "parking", "Shoutout to whoever fixed the parking gate, no more 10 minute queues!"),
    ("en", "negative", "parking", "Parking is impossible after 8am. I'm not even joking."),
    ("en", "neutral", "events", "Career Days start Wednesday in the main hall, 10am to 4pm."),
    ("en", "positive", "events", "Lipa Fest was amazing, best night of the semester 🎶"),
    ("en", "negative", "cafeteria", "Found a hair in my soup at the cafeteria. Not ordering that again."),
    ("en", "positive", "exams", "Passed! Not bad for someone who started studying two days before 😅"),  # negation, but positive
    ("en", "neutral", "wifi", "How do I connect to LipaNet on my phone? The guide only covers laptops."),
    ("en", "negative", "exams", "Oh wonderful, the exam schedule changed again. Love that for us."),  # sarcasm
    ("en", "positive", "professors", "Not gonna lie, I didn't expect to like this course, but {prof} made it fun."),
    ("en", "neutral", "cafeteria", "Is the cafeteria open during the winter break?"),
    ("en", "negative", "wifi", "wifi in dorm 2 is so slow i could walk to brezovac faster lol"),
    ("en", "positive", "enrollment", "Enrollment took 5 minutes this year, honestly impressed"),
    # DEBATABLE sentiment
    ("en", "neutral", "exams", "Well, that was an exam."),
    ("en", "negative", "professors", "{prof}'s lectures are certainly... an experience."),
    ("en", "positive", "events", "Can't say I hated the karaoke night 😂"),
    ("en", "neutral", "cafeteria", "The cafeteria changed the menu. Bold move."),
    # Very short posts
    ("en", "positive", "other", "nice"),
    ("en", "negative", "other", "ugh"),
    ("en", "neutral", "other", "same"),
    ("en", "neutral", "other", "meh"),  # neutral or mildly negative? Debatable.
    ("en", "positive", "other", "yay"),
    ("en", "neutral", "other", "noted"),
    # More everyday posts
    ("en", "negative", "wifi", "The dorm wifi drops every evening at exactly 9pm. Every single evening."),
    ("en", "positive", "wifi", "Whoever upgraded the routers in the reading room, thank you, it's finally usable."),
    ("en", "neutral", "wifi", "Is there a separate wifi for guests, or do they use LipaNet-Gost?"),
    ("en", "negative", "wifi", "Can't even open the course page from the lab, the network keeps blocking it."),
    ("en", "negative", "cafeteria", "Paid for a full meal and got three spoons of rice. Seriously?"),
    ("en", "positive", "cafeteria", "The Friday soup at the cafeteria is better than my mom's. Don't tell her."),
    ("en", "neutral", "cafeteria", "Does the cafeteria have gluten-free options?"),
    ("en", "neutral", "cafeteria", "What time does the cafeteria stop serving breakfast?"),
    ("en", "negative", "exams", "Our midterm grades still aren't out after a month. How are we supposed to plan?"),
    ("en", "positive", "exams", "Got an A on the programming project, all those late nights paid off!"),
    ("en", "neutral", "exams", "Can we bring a calculator to the statistics final?"),
    ("en", "negative", "exams", "Two exams on the same morning in different buildings. Brilliant planning."),  # sarcasm
    ("en", "negative", "exams", "Retake fee went up again. We're students, not banks."),
    ("en", "positive", "professors", "{prof} remembered my name after one class. Small thing, but it made my day."),
    ("en", "negative", "professors", "{prof} reads the slides word for word for ninety minutes. I could do that at home."),
    ("en", "neutral", "professors", "Does anyone know {prof}'s office hours this week?"),
    ("en", "positive", "professors", "Best lecture of the year today, {prof} brought actual robots to class."),
    ("en", "negative", "enrollment", "The student office lost my documents. Twice."),
    ("en", "positive", "enrollment", "The new online enrollment form is so much easier than the paper one."),
    ("en", "neutral", "enrollment", "Where do I pick up my student ID card?"),
    ("en", "negative", "parking", "Got a parking ticket while unloading boxes for five minutes. Five."),
    ("en", "positive", "parking", "Free parking during exam week is the best decision the university has made."),
    ("en", "neutral", "parking", "Is the parking lot behind the library open on Sundays?"),
    ("en", "positive", "events", "The film night in the dorm was so cozy, we should do it every month."),
    ("en", "negative", "events", "The career fair had five companies and all of them were hiring sales reps."),
    ("en", "neutral", "events", "Volunteers needed for Lipa Fest setup on Thursday, message the student union."),
    ("en", "positive", "events", "First time at the chess tournament and I won two games!"),
    ("en", "negative", "other", "The heating in building C is broken and it's 12 degrees in the lecture hall."),
    ("en", "positive", "other", "Shoutout to the cleaning staff, the bathrooms in building A are spotless now."),
    ("en", "neutral", "other", "Anyone found a black backpack in room 101? It's mine."),

    # ---------------- Mixed (code-switching) ----------------
    ("mixed", "negative", "exams", "exam je bio insane, ne mogu vise"),
    ("mixed", "positive", "events", "Lipa Fest je bio literally the best, vidimo se next year 🔥"),
    ("mixed", "negative", "wifi", "wifi je down opet, I swear ovo je najgori faks"),
    ("mixed", "neutral", "enrollment", "jel neko zna deadline za upis? on the website ne pise nista"),
    ("mixed", "positive", "professors", "{prof} je honestly legend, objasnio mi sve u 10 minuta"),
    ("mixed", "negative", "cafeteria", "menza food je bio cold again, I'm so done"),
    ("mixed", "neutral", "parking", "ima li free parking negdje blizu kampusa? asking for a friend"),
    ("mixed", "positive", "exams", "passed sam!!! finally mogu spavat 😭"),
    ("mixed", "negative", "professors", "{prof} nam dao assignment za vikend, thanks a lot 🙃"),  # sarcasm
    ("mixed", "neutral", "events", "ko ide na career fair sutra? need company"),
    ("mixed", "positive", "cafeteria", "burek u menzi danas was actually fire 🔥"),
    ("mixed", "negative", "enrollment", "eLipa je crashed na zadnji dan prijave, classic"),
    # DEBATABLE sentiment
    ("mixed", "neutral", "exams", "kolokvij je bio ok-ish, not sure kako sam prosao"),
    ("mixed", "negative", "wifi", "novi wifi je... interesting, let's say"),
    # More everyday posts
    ("mixed", "negative", "wifi", "LipaNet je spor as hell danas, can't even load eLipa"),
    ("mixed", "positive", "wifi", "novi router u labu is a game changer, sve radi"),
    ("mixed", "neutral", "wifi", "zna li neko password za guest wifi, need it asap"),
    ("mixed", "negative", "cafeteria", "ručak u menzi je bio so bad da sam otišao na burek"),
    ("mixed", "positive", "cafeteria", "pancakes u menzi subotom su top tier, highly recommend"),
    ("mixed", "neutral", "cafeteria", "jel menza open tokom praznika or not?"),
    ("mixed", "negative", "exams", "kolokvij iz statistike was a nightmare, pola nas nije ni završilo"),
    ("mixed", "positive", "exams", "dobio sam desetku iz programiranja, let's goooo"),
    ("mixed", "neutral", "exams", "kad je deadline za seminarski, anyone?"),
    ("mixed", "negative", "professors", "{prof} je opet cancel-ovao class bez warninga, typical"),
    ("mixed", "positive", "professors", "{prof} explains stvari tako jasno, finally razumijem rekurziju"),
    ("mixed", "negative", "enrollment", "studentska služba lost my papers, opet moram sve ispočetka"),
    ("mixed", "positive", "enrollment", "upis je bio super smooth ove godine, no lines at all"),
    ("mixed", "neutral", "parking", "ima li neko parking spot za iznajmiti near kampusa?"),
    ("mixed", "positive", "events", "karaoke night u domu was amazing, ekipa je bila top"),
    ("mixed", "negative", "events", "career fair je bio total waste of time, samo dvije firme"),
]

# Posts whose SENTIMENT gold label is debatable: sarcasm that might be sincere,
# understatement, "Bold move". Reasonable people can disagree about them.
# They get sentiment_debatable=True, and we do not make variants of them.
DEBATABLE_SENTIMENT_TEXTS = {
    "Ispit je bio... zanimljiv.",
    "Ma {prof} je poseban lik, to ću ti samo reći.",
    "Menza danas iznenadila. Ne znam još jel to dobro.",
    "Upisala sam se. Sad tek počinje.",
    "{prof} je danas bio drugačiji nego inače.",
    "Bilo je bolje nego prošle godine, ali to i nije teško.",
    "Nije mi jasno je li {prof} rekao da je kolokvij obavezan ili nije.",
    "Nije loše za te novce, ručak u menzi je skroz ok.",
    "{prof} je rekao da će ispit biti \"lak\". Videćemo.",
    "Dobio sam šesticu. Prošao sam, valjda je to to.",
    "Well, that was an exam.",
    "{prof}'s lectures are certainly... an experience.",
    "Can't say I hated the karaoke night 😂",
    "The cafeteria changed the menu. Bold move.",
    "kolokvij je bio ok-ish, not sure kako sam prosao",
    "novi wifi je... interesting, let's say",
    "ma super",
    "meh",
}

# Posts that are genuinely UNDECIDABLE between Bosnian and Croatian. Their
# lang_gold is "bs", but they get ambiguous=True: a detector that says "hr"
# is not really wrong. We do not make variants of them.
AMBIGUOUS_LANGUAGE_TEXTS = {
    "Rezultati kolokvija bit će objavljeni na oglasnoj ploči.",
    "Predavanje iz statistike danas počinje pola sata kasnije.",
    "Vidimo se ispred zgrade B poslije predavanja.",
    "Danas opet nema vježbi, a o tome nismo ništa znali.",
    "Imamo li sutra nastavu ili je odgođena?",
    "Ispit je pomjeren za ponedjeljak u devet.",
    "Parking iza zgrade A je cijeli dan zatvoren, a nigdje nema objave.",
    "Lijep dan za učenje vani, cijela ekipa je na travi ispred fakulteta.",
    "Dvorana je bila prepuna, pola nas je pisalo ispit na koljenima.",
    "{prof} je danas objasnio gradivo bolje nego cijela knjiga.",
}

DIACRITICS_MAP = str.maketrans({
    "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
    "Č": "C", "Ć": "C", "Š": "S", "Ž": "Z", "Đ": "Dj",
})
BCS_LANGS = {"bs", "hr", "sr"}
SPLITS = ["dev", "test"]


# --- Small text helpers ------------------------------------------------------

def strip_diacritics(text):
    """Write č ć š ž đ as c c s z dj, the way many people type on phones."""
    return text.translate(DIACRITICS_MAP)


def has_diacritics(text):
    return strip_diacritics(text) != text


def fill_professor(text, professor, lang):
    """Put a professor's name into the {prof} slot."""
    # English speakers usually type names without diacritics.
    if lang in ("en", "mixed"):
        professor = strip_diacritics(professor)
    # A post that starts with the name should start with a capital letter.
    if text.startswith("{prof}"):
        professor = professor[0].upper() + professor[1:]
    return text.replace("{prof}", professor)


def add_typo(text, rng):
    """Make one realistic typo: swap, drop or double a letter in a longer word."""
    words = text.split(" ")
    long_word_positions = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not long_word_positions:
        return text
    i = rng.choice(long_word_positions)
    word = words[i]
    j = rng.randrange(1, len(word) - 2)  # never touch the first letter
    kind = rng.choice(["swap", "drop", "double"])
    if kind == "swap":
        word = word[:j] + word[j + 1] + word[j] + word[j + 2:]
    elif kind == "drop":
        word = word[:j] + word[j + 1:]
    else:
        word = word[:j] + word[j] + word[j:]
    words[i] = word
    return " ".join(words)


def make_casual(text):
    """Chat style: all lowercase and no final full stop."""
    return text.lower().rstrip(".")


# --- Building the corpus -----------------------------------------------------

def make_base_rows():
    """Turn BASE_POSTS into rows. Each base post gets one professor name."""
    rows = []
    for base_id, (lang, sentiment, topic, template) in enumerate(BASE_POSTS):
        professor = PROFESSORS[base_id % len(PROFESSORS)]
        rows.append({
            "base_id": base_id,
            "template": template,
            "lang_gold": lang,
            "sentiment_gold": sentiment,
            "topic": topic,
            "ambiguous": template in AMBIGUOUS_LANGUAGE_TEXTS,
            "sentiment_debatable": template in DEBATABLE_SENTIMENT_TEXTS,
            "text": fill_professor(template, professor, lang),
        })
    return rows


def can_have_variants(row):
    # Variants of "ok" or of a flagged post would be pointless duplicates.
    is_short = len(row["template"].split()) <= 3
    is_flagged = row["ambiguous"] or row["sentiment_debatable"]
    return not is_short and not is_flagged


def make_variant(row, rng):
    """The same post typed again: mostly WITHOUT diacritics, sometimes with a typo.

    We never append "lol", "haha" or emoji: a real student would not add a
    laughing emoji to a complaint, and such copies only inflate the corpus.
    """
    text = row["text"]
    if "{prof}" in row["template"] and rng.random() < 0.3:
        # The same complaint about a different professor.
        text = fill_professor(row["template"], rng.choice(PROFESSORS), row["lang_gold"])
    if has_diacritics(text):
        text = strip_diacritics(text)
    else:
        text = add_typo(text, rng)
    # Sometimes also typed in lowercase chat style.
    if rng.random() < 0.3:
        text = make_casual(text)
    return {**row, "text": text}


def add_variants(rows, rng):
    """Add a FEW variants until every language reaches its target count.

    Almost all posts are different base posts. Variants are kept rare on
    purpose: near-duplicates make a corpus look bigger than it is, and a copy
    in the dev half "leaks" the answer for its twin in the test half.
    """
    seen_texts = {row["text"] for row in rows}
    all_rows = list(rows)
    for lang, target in TARGET_PER_LANG.items():
        sources = [r for r in rows if r["lang_gold"] == lang and can_have_variants(r)]
        rng.shuffle(sources)
        count = sum(1 for r in all_rows if r["lang_gold"] == lang)
        i = 0
        while count < target:
            # Each base post gets at most one variant.
            variant = make_variant(sources[i], rng)
            i += 1
            if variant["text"] in seen_texts:
                continue  # exact duplicate - try the next post
            seen_texts.add(variant["text"])
            all_rows.append(variant)
            count += 1
    return all_rows


def remove_some_diacritics(rows, rng):
    """Strip diacritics from random BCS posts until ~40% of them have none."""
    bcs_rows = [r for r in rows if r["lang_gold"] in BCS_LANGS]
    target = round(NO_DIACRITICS_SHARE * len(bcs_rows))
    already_plain = sum(1 for r in bcs_rows if not has_diacritics(r["text"]))
    candidates = [r for r in bcs_rows if has_diacritics(r["text"])]
    rng.shuffle(candidates)
    seen_texts = {r["text"] for r in rows}
    for row in candidates:
        if already_plain >= target:
            break
        plain = strip_diacritics(row["text"])
        if plain in seen_texts:
            continue  # would duplicate another post
        seen_texts.add(plain)
        row["text"] = plain
        already_plain += 1


def assign_splits(rows, rng):
    """Put every post in "dev" (50%) or "test" (50%), separately per language.

    A post and its variant always land in the SAME half. Otherwise a rule
    tuned on the dev copy would be tested on its near-identical twin.
    """
    for lang in TARGET_PER_LANG:
        # Group the posts of this language by base post.
        groups = {}
        for row in rows:
            if row["lang_gold"] == lang:
                groups.setdefault(row["base_id"], []).append(row)
        group_list = list(groups.values())
        rng.shuffle(group_list)

        # Give each group to the half that has fewer posts so far.
        sizes = {"dev": 0, "test": 0}
        for group in group_list:
            split = "dev" if sizes["dev"] <= sizes["test"] else "test"
            for row in group:
                row["split"] = split
            sizes[split] += len(group)


def order_rows(rows, rng):
    """Shuffle, but keep the first rows free of variants of the same post."""
    rng.shuffle(rows)
    first, rest, used_base_ids = [], [], set()
    for row in rows:
        if len(first) < FIRST_ROWS_WITHOUT_VARIANTS and row["base_id"] not in used_base_ids:
            first.append(row)
            used_base_ids.add(row["base_id"])
        else:
            rest.append(row)
    return first + rest


def random_date(start, end, rng):
    """A random day between start and end (both included)."""
    return start + timedelta(days=rng.randint(0, (end - start).days))


def pick_date(row, rng):
    """The posting date: inside a planted event window, or anywhere in the year."""
    for event in EVENTS:
        matches = row["topic"] == event["topic"] and row["sentiment_gold"] == event["sentiment"]
        if matches and rng.random() < event["share"]:
            return random_date(event["start"], event["end"], rng)
    return random_date(START_DATE, END_DATE, rng)


def pick_title(source, topic, lang, rng):
    if source == "Forum Lipa":
        titles = FORUM_TITLES_EN if lang == "en" else FORUM_TITLES_BCS
        return rng.choice(titles[topic])
    if source == "Novine Brezovac komentari":
        return NEWS_HEADLINES[topic][0]
    return ""  # chat messages and confessions have no title


def to_csv_records(rows, rng):
    records = []
    for post_id, row in enumerate(rows, start=1):
        source = rng.choices(SOURCES, weights=SOURCE_WEIGHTS)[0]
        records.append({
            "id": post_id,
            "source": source,
            "url": f"https://example.com/post/{post_id}",
            "date": pick_date(row, rng).isoformat(),
            "title": pick_title(source, row["topic"], row["lang_gold"], rng),
            "text": row["text"],
            "lang": "",  # left empty on purpose: the pipeline fills it in
            "lang_gold": row["lang_gold"],
            "sentiment_gold": row["sentiment_gold"],
            "topic_gold": row["topic"],
            "ambiguous": row["ambiguous"],
            "sentiment_debatable": row["sentiment_debatable"],
            "split": row["split"],
        })
    return records


def check(rows, records):
    """Fail loudly if the corpus breaks one of our own rules."""
    texts = [r["text"] for r in records]
    assert len(texts) == len(set(texts)), "duplicate texts found"
    first_base_ids = [r["base_id"] for r in rows[:FIRST_ROWS_WITHOUT_VARIANTS]]
    assert len(first_base_ids) == len(set(first_base_ids)), "near-duplicates in first rows"
    assert all("{prof}" not in t for t in texts), "unfilled {prof} slot"
    for lang, target in TARGET_PER_LANG.items():
        actual = sum(1 for r in records if r["lang_gold"] == lang)
        assert actual == target, f"{lang}: expected {target}, got {actual}"
        in_dev = sum(1 for r in records if r["lang_gold"] == lang and r["split"] == "dev")
        assert abs(2 * in_dev - target) <= 2, f"{lang}: dev/test split is not 50/50"
    assert len(DEBATABLE_SENTIMENT_TEXTS & AMBIGUOUS_LANGUAGE_TEXTS) == 0
    flagged = sum(1 for r in records if r["ambiguous"])
    assert flagged == len(AMBIGUOUS_LANGUAGE_TEXTS), "ambiguous post missing or copied"


def main():
    rng = random.Random(SEED)
    rows = make_base_rows()
    rows = add_variants(rows, rng)
    remove_some_diacritics(rows, rng)
    assign_splits(rows, rng)
    rows = order_rows(rows, rng)
    records = to_csv_records(rows, rng)
    check(rows, records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    variants = len(records) - len(BASE_POSTS)
    print(f"Wrote {len(records)} SYNTHETIC posts to {OUTPUT_PATH}")
    print(f"  {len(BASE_POSTS)} different base posts + {variants} variants")


if __name__ == "__main__":
    main()
