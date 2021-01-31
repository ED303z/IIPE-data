import os
import pandas as pd
import nltk
import requests
import bs4
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from IIPE.constants import ALL_STOP_WORDS


def make_contents_df(lst):
    """Returns a dataframe with date, reference, text from a list of file_names"""
    # init a list of dictionnaries
    ld_contents = []

    for file in tqdm(lst):
        if file.endswith(".txt"):
            # keeping the reference and the date
            split = (
                file.replace("Reports_Plain text_", "").replace(".txt", "").split("_")
            )
            reference = split[0]
            date = "-".join(split[1:][::-1])

            # creating the dictionnary
            d = {"date": date, "reference": reference, "text": ""}

            # adding text content to the dictionary
            with open(file, encoding="utf8", errors="ignore") as f:
                text = f.read()
                d["text"] = text
            ld_contents.append(d)

    # create dataframe and set date to a datetime datatype
    df_contents = pd.DataFrame(ld_contents)
    df_contents["date"] = pd.to_datetime(df_contents["date"])
    return df_contents


def make_tokens(df):
    """Removes stopwords, stems and lemmatizes
    Returns clean tokens"""

    stopwords = set(nltk.corpus.stopwords.words("english"))

    # turns the text in the dataframe into a long list of words
    TotalText = list(df.text.values)

    # stopwords, with plurals (otherwise the lemmatizong steps puts some of the stopwords back)
    stopwords = stopwords.union(ALL_STOP_WORDS)
    TotalText = " ".join(TotalText)

    # tokenization
    tokens = [
        w for w in word_tokenize(TotalText.lower()) if w.isalpha()
    ]  # isalpha() checks if each word is alphabetical, lower() transforms everything to lowercase
    no_stop = [
        t.strip() for t in tokens if t.strip() not in stopwords
    ]  # stopwords already comes with a built-in list of words to remove
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stop]

    return lemmatized


def get_inspection_reports(local_file=False):
    """Code provided to scrape tthe education.ie website. We taked the first 142 pages info (where PDFs are located)
    to return a dataframe with details about the report ['Date', 'School Roll No.', 'County', 'School Name', 'School Level',
       'Inspection Type', 'Subject, 'URL']"""

    # Prefer the use of a saved csv, as the scraping takes time
    # After 1st scraping => update csv_path & filename accordingly
    # For the hackathon, meant to be used with Google Colab
    if local_file:
        csv_path = os.path.join("..", "..", "IIPE-colab", "data", "csv")
        return pd.read_csv(os.path(csv_path, "General_InspectionReports.csv")).drop(
            columns=["Unnamed: 0"]
        )

    # Scraping
    WebpageRoot = "https://www.education.ie/en/Publications/Inspection-Reports-Publications/Whole-School-Evaluation-Reports-List/?pageNumber="
    General_InspectionReports = pd.DataFrame(
        columns=[
            "Date",
            "School Roll No.",
            "County",
            "School Name",
            "School Level",
            "Inspection Type",
            "Subject",
            "URL",
        ]
    )

    for x in range(1, 143):
        IrelandWebpage = requests.get(WebpageRoot + str(x))
        CleanIrelandWebpage = bs4.BeautifulSoup(IrelandWebpage.text, "lxml")
        InspectionReports = {}
        ID = 0
        Table = CleanIrelandWebpage.find("table", id="IRList")
        for p in Table.find_all("tr"):
            if ID == 0:
                ID = ID + 1
                continue
            else:
                Date = (
                    p("td")[0].string[:2]
                    + "_"
                    + p("td")[0].string[3:5]
                    + "_"
                    + p("td")[0].string[6:]
                )
                SchoolRoll = p("td")[1].string
                County = p("td")[2].string
                SchoolName = p("td")[3].string
                SchoolLevel = p("td")[4].string
                InspectionType = p("td")[5].string
                Subject = p("td")[6].string
                URL = p("td")[7]("a")[0].attrs["href"][86:]
                InspectionReports[ID] = {
                    "Date": Date,
                    "School Roll No.": SchoolRoll,
                    "County": County,
                    "School Name": SchoolName,
                    "School Level": SchoolLevel,
                    "Inspection Type": InspectionType,
                    "Subject": Subject,
                    "URL": URL,
                }
                ID = ID + 1

        # Dataframe creation
        df_InspectionReports = pd.DataFrame.from_dict(InspectionReports, orient="index")
        General_InspectionReports = pd.concat(
            [General_InspectionReports, df_InspectionReports]
        )
        return General_InspectionReports


def make_contents_details_df(df_contents, gen_ins_report):
    """Joins two dataframes to enable an analysis by date and/or by county"""

    # Drop Subject (all NaN) and rename columns for the merge
    gen_ins_report = gen_ins_report.drop(columns=["Subject"]).rename(
        columns={"School Roll No.": "reference"}
    )

    return df_contents.merge(gen_ins_report, on="reference")


def date_processed_df(df):
    """Re-order `Date` string from scraping e.g. '11-04-2014' => '2014-04-11', then convert to datetime"""
    df["Date"] = df.Date.apply(lambda x: "-".join(x.split("_")[::-1]))
    df["date_match"] = df["date"] == df["Date"]
    df["date"] = pd.to_datetime(df.date)
    df["Date"] = pd.to_datetime(df.date)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


def cont_det_by_counties(df):
    """Creates a dictionary of dataframes, one per county"""
    counties_df_dict = {}
    for county in df.County.unique():
        df_county = df[df["County"] == f"{county}"]
        counties_df_dict[county] = df
    return counties_df_dict


if __name__ == "__main__":
    os.chdir(os.path.join("data_sample", "plain_text_sample"))
    # print(os.getcwd())
    # print("Files to be processed : ", os.listdir())

    df = make_contents_df(os.listdir())
    # print(df)
    tokens = make_tokens(df)
    # print(tokens)

    # TO DO : convert the prints below into unit tests
    # print(df.shape)
    # print(df.dtypes)
    # get_inspection_reports(local_file=True) => refactor with try: except:
    # print(f"We have {len(tokens)} tokens")
