#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, sqlite3


class DB(object):
    """
	This class provides functions for DB operations
	with exception handling.
	"""

    def __init__(self, dbFile):
        try:
            conn = sqlite3.connect(dbFile)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
        except:
            print("Error opening {}".format(dbFile))
            sys.exit(1)

        self.conn = conn
        self.c = c

    def select(self, cmd):
        try:
            self.c.execute(cmd)
            #rows = self.c.fetchall()
            rows = [dict(row) for row in self.c.fetchall()]
        except sqlite3.Error as e:
            print(e)
            sys.exit(1)
        except:
            print("Uncaught error in SQLite access while executing {}".format(cmd))
            sys.exit(1)

        #print(rows)
        return rows

    def insert(self, cmd, data):
        try:
            self.c.execute(cmd, data)
        except sqlite3.Error as e:
            print(e)
            sys.exit(1)
        except:
            print("Uncaught error in SQLite access while executing {}".format(cmd))
            sys.exit(1)

    def execute(self, cmd):
        try:
            self.c.execute(cmd)
        except sqlite3.Error as e:
            print(e)
            sys.exit(1)
        except:
            print("Uncaught error in SQLite access while executing {}".format(cmd))
            sys.exit(1)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.c.close()
        self.conn.close()
