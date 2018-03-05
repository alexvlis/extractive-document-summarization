		DUC 2001 Summarization Documents


Created: March 1, 2011


1. Overview

This package contains trial, training, and test data from DUC 2001.

Additional DUC 2001 data are available at:
   http://duc.nist.gov/data.html

DUC 2001 task guidelines are available at:
   http://duc.nist.gov/guidelines/2001.html


2. Contents


2.1 data/trial/

The trial summary data was prepared for discussion at DUC 2000 - an
early version of data to be used for DUC 2001. Note that the tagging
will likely change and that the summaries have not been
proofread. Only three sets are complete. "pres92" contains only the
400- and 200-word summaries. The data directory is password protected
since access to the TREC data requires that a signed TREC data
permission form be on file at NIST.

The trial summarization file sets each comprise:

    * WSJ*: a set of 10 TREC (Wall Street Journal) documents each
    * assessor_x (subdirectory with assessor x's summaries)
          o summaries: 10 100-word summaries, one per document
          o 400: a 400-word multi-document summary of all the documents
          o 200: a 200-word multi-document summary of all the documents
          o 100: a 100-word multi-document summary of all the documents
          o 50: a 50-word multi-document summary of all the documents
    * assessor_y (subdirectory with assessor y's summaries)
    * ...


2.2 data/training/

The training data consist of a randomly selected sample for each of
the 10 NIST assessors of 3 document sets with all their associated
summaries. For each set the directory/file structure is laid out as
follows:

    * dnnx:   document set number nn (01-60) selected by assessor x
          o docs: full text of the documents in this set
          o dnnxy: summaries created by NIST assessor y for document set dnn (y==x)
                + perdocs:   all the 100-word single-document summaries
                + 400:   a 400-word multi-document summary of all the documents
                + 200:   a 200-word multi-document summary of all the documents
                + 100:   a 100-word multi-document summary of all the documents
                + 50:   a 50-word multi-document summary of all the documents

Additional files include the following:

    * summaries.dtd - a DTD to which all of the trial summaries conform
    * duc2001training.tar.gz - a gzipped tar file of all the training data


2.3 data/test/

The test data consist of a randomly selected sample for each of the 10
NIST assessors of 3 document sets. For each set the directory/file
structure is laid out as follows:

    * docs
          o dnnx:   document set number nn (01-60) selected by assessor x
                + full text of the documents in this set
                  Note: FBIS documents are not browsable - view source
          o duc2001testdocs.tar.gz - a gzipped tar file of all the test data

    * original.summaries
          o dnnxy:   document set number nn (01-60) selected by assessor x and summarized by assessor y. For original summaries x == y.
            Note: the original multi-document summaries for d31 were not used in the evaluation and are not included here because through a misunderstanding they were based on a slightly different set of documents than was distributed.
                + 100   (the 100-word multi-document summary)
                + 200   (the 200-word multi-document summary)
                + 400   (the 400-word multi-document summary)
                + 50     (the 50-word multi-document summary)
                + perdocs   (the single-document summaries in one file)
          o duc2001testorigsum.tar.gz - a gzipped tar file of the original summaries

    * duplicate.summaries
          o dnnxy:   document set number nn (01-60) selected by assessor x and summarized by assessor y. For duplicate summaries, the selector is a different person from the summarizer.
                + 100
                + 200
                + 400
                + 50
                + perdocs
          o duc2001testdupsum.tar.gz - a gzipped tar file of the duplicate summaries


2.4 data/testtraining/

The file Duc2001testtraining.tar.gz was kindly provided by Terry
Copeck at the University of Ottawa for DUC 2002. 

It contains the usual two-level topic/document/ directory
organization.  Within each document directory are three files:
'<doc>.abs' and '<doc>.body' contain respectively the assessor's
summary and the document body (with none of the tagged data that lie
outside the body and that have come up in recent list discussion).  A
third '<doc>.txt' file contains a concatenation of abstract and body
prefixed by 'Abstract:' and 'Introduction:' headings (for systems that
want to take the document apart themselves).

All files are plain vanilla ASCII with DOS line endings (the HTML tags
were stripped out of the summaries on a PC).


3. Contact Information

For further information about the contents of this data release,
please contact:

  Hoa Dang, NIST, TAC/DUC organizer                <hoa.dang@nist.gov>
