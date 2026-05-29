"""
utils/db.py
-----------
MongoDB data layer — replaces data/movies.csv and data/interactions.csv.

Collections
-----------
  recsys.movies        — one document per movie
  recsys.interactions  — one document per user-movie rating

Schema
------
movies:
  { movie_id: int, title: str, genre: str, year: int, description: str }

interactions:
  { user_id: int, movie_id: int, rating: float }

Usage
-----
    from utils.db import MongoDataLoader
    loader = MongoDataLoader()          # default: localhost:27017
    movies_df, interactions_df = loader.load()

Seeding (first run)
-------------------
    loader.seed()   # inserts sample data if collections are empty
"""

import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure


# ---------------------------------------------------------------------------
# Sample data — mirrors the original CSV files
# ---------------------------------------------------------------------------

MOVIES = [
    {"movie_id": 1,  "title": "Interstellar",       "genre": "sci-fi",  "year": 2014, "description": "A team of astronauts travels through a wormhole in search of a new home for humanity beyond our dying planet."},
    {"movie_id": 2,  "title": "The Martian",         "genre": "sci-fi",  "year": 2015, "description": "An astronaut stranded on Mars must use his ingenuity to survive while NASA engineers race to bring him home."},
    {"movie_id": 3,  "title": "Gravity",             "genre": "sci-fi",  "year": 2013, "description": "Two astronauts struggle to survive in space after their shuttle is destroyed by debris."},
    {"movie_id": 4,  "title": "Inception",           "genre": "sci-fi",  "year": 2010, "description": "A thief who steals corporate secrets through dream-sharing technology is tasked with planting an idea into a CEO's mind."},
    {"movie_id": 5,  "title": "Arrival",             "genre": "sci-fi",  "year": 2016, "description": "A linguist is recruited by the military to communicate with alien lifeforms after a dozen mysterious spacecraft appear around the world."},
    {"movie_id": 6,  "title": "Ex Machina",          "genre": "sci-fi",  "year": 2014, "description": "A programmer is selected to participate in a groundbreaking experiment in synthetic intelligence by evaluating a humanoid robot."},
    {"movie_id": 7,  "title": "The Notebook",        "genre": "romance", "year": 2004, "description": "A young couple falls in love during the 1940s, but their romance is threatened by class differences and family opposition."},
    {"movie_id": 8,  "title": "Pride and Prejudice", "genre": "romance", "year": 2005, "description": "Sparks fly when spirited Elizabeth Bennet meets the proud and arrogant Mr. Darcy in Regency-era England."},
    {"movie_id": 9,  "title": "La La Land",          "genre": "romance", "year": 2016, "description": "A jazz musician and an aspiring actress fall in love while pursuing their dreams in Los Angeles."},
    {"movie_id": 10, "title": "Titanic",             "genre": "romance", "year": 1997, "description": "A fictional love story set against the backdrop of the ill-fated maiden voyage of the RMS Titanic."},
    {"movie_id": 11, "title": "The Dark Knight",     "genre": "action",  "year": 2008, "description": "Batman battles the Joker, a criminal mastermind who plunges Gotham City into anarchy using a campaign of chaos and terror."},
    {"movie_id": 12, "title": "Mad Max Fury Road",   "genre": "action",  "year": 2015, "description": "In a post-apocalyptic wasteland, Max teams with a rebel warrior to flee from a cult leader controlling the water supply."},
    {"movie_id": 13, "title": "John Wick",           "genre": "action",  "year": 2014, "description": "A retired hitman seeks vengeance against the Russian mobster who killed his dog and stole his car."},
    {"movie_id": 14, "title": "The Matrix",          "genre": "action",  "year": 1999, "description": "A computer hacker discovers that reality is a simulation and joins a rebellion against the machines that control it."},
    {"movie_id": 15, "title": "Avengers Endgame",    "genre": "action",  "year": 2019, "description": "The Avengers assemble to reverse the damage caused by Thanos and restore balance to the universe."},
]

INTERACTIONS = [
    {"user_id": 1, "movie_id": 1,  "rating": 5},
    {"user_id": 1, "movie_id": 2,  "rating": 4},
    {"user_id": 1, "movie_id": 4,  "rating": 5},
    {"user_id": 1, "movie_id": 6,  "rating": 4},
    {"user_id": 2, "movie_id": 1,  "rating": 4},
    {"user_id": 2, "movie_id": 3,  "rating": 5},
    {"user_id": 2, "movie_id": 5,  "rating": 5},
    {"user_id": 2, "movie_id": 6,  "rating": 3},
    {"user_id": 3, "movie_id": 7,  "rating": 5},
    {"user_id": 3, "movie_id": 8,  "rating": 4},
    {"user_id": 3, "movie_id": 9,  "rating": 5},
    {"user_id": 3, "movie_id": 10, "rating": 4},
    {"user_id": 4, "movie_id": 11, "rating": 5},
    {"user_id": 4, "movie_id": 12, "rating": 4},
    {"user_id": 4, "movie_id": 13, "rating": 5},
    {"user_id": 4, "movie_id": 14, "rating": 5},
    {"user_id": 5, "movie_id": 1,  "rating": 3},
    {"user_id": 5, "movie_id": 7,  "rating": 4},
    {"user_id": 5, "movie_id": 11, "rating": 5},
    {"user_id": 5, "movie_id": 14, "rating": 4},
]


# ---------------------------------------------------------------------------
# MongoDataLoader
# ---------------------------------------------------------------------------

class MongoDataLoader:
    """
    Reads movies and interactions from MongoDB and returns pandas DataFrames.

    Parameters
    ----------
    uri : str
        MongoDB connection URI.  Default: mongodb://localhost:27017
    db_name : str
        Database name.  Default: recsys
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "recsys",
    ):
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")           # verify connection
        except ConnectionFailure as e:
            raise ConnectionError(
                f"[MongoDB] Cannot connect to {uri}. "
                f"Is MongoDB running?\n  {e}"
            )

        self.db = self.client[db_name]
        self.movies_col      = self.db["movies"]
        self.interactions_col = self.db["interactions"]
        print(f"[MongoDB] Connected — database: '{db_name}'")

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------

    def seed(self, force: bool = False) -> None:
        """
        Insert sample data if the collections are empty (or force=True).

        Parameters
        ----------
        force : bool
            If True, drop existing data and re-insert.
        """
        if force:
            self.movies_col.drop()
            self.interactions_col.drop()
            print("[MongoDB] Collections dropped (force=True).")

        if self.movies_col.count_documents({}) == 0:
            self.movies_col.insert_many(MOVIES)
            # Index on movie_id for fast lookups
            self.movies_col.create_index([("movie_id", ASCENDING)], unique=True)
            print(f"[MongoDB] Seeded {len(MOVIES)} movies.")
        else:
            print(f"[MongoDB] movies collection already has "
                  f"{self.movies_col.count_documents({})} documents — skipping seed.")

        if self.interactions_col.count_documents({}) == 0:
            self.interactions_col.insert_many(INTERACTIONS)
            # Compound index for fast user/movie queries
            self.interactions_col.create_index(
                [("user_id", ASCENDING), ("movie_id", ASCENDING)]
            )
            print(f"[MongoDB] Seeded {len(INTERACTIONS)} interactions.")
        else:
            print(f"[MongoDB] interactions collection already has "
                  f"{self.interactions_col.count_documents({})} documents — skipping seed.")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load movies and interactions from MongoDB into DataFrames.

        Returns
        -------
        movies_df : pd.DataFrame
            Columns: movie_id, title, genre, year, description
        interactions_df : pd.DataFrame
            Columns: user_id, movie_id, rating
        """
        movies_docs = list(self.movies_col.find({}, {"_id": 0}))
        inter_docs  = list(self.interactions_col.find({}, {"_id": 0}))

        if not movies_docs:
            raise RuntimeError(
                "[MongoDB] 'movies' collection is empty. Run loader.seed() first."
            )
        if not inter_docs:
            raise RuntimeError(
                "[MongoDB] 'interactions' collection is empty. Run loader.seed() first."
            )

        movies_df       = pd.DataFrame(movies_docs)
        interactions_df = pd.DataFrame(inter_docs)

        print(f"[MongoDB] Loaded {len(movies_df)} movies, "
              f"{len(interactions_df)} interactions.")
        return movies_df, interactions_df

    # ------------------------------------------------------------------
    # Helpers — optional CRUD for extending the system
    # ------------------------------------------------------------------

    def add_movie(self, movie: dict) -> None:
        """Insert or update a single movie document."""
        self.movies_col.update_one(
            {"movie_id": movie["movie_id"]},
            {"$set": movie},
            upsert=True,
        )

    def add_interaction(self, user_id: int, movie_id: int, rating: float) -> None:
        """Insert or update a single user-movie rating."""
        self.interactions_col.update_one(
            {"user_id": user_id, "movie_id": movie_id},
            {"$set": {"rating": rating}},
            upsert=True,
        )

    def get_user_ratings(self, user_id: int) -> list[dict]:
        """Return all ratings for a given user."""
        return list(
            self.interactions_col.find({"user_id": user_id}, {"_id": 0})
        )

    def close(self) -> None:
        self.client.close()