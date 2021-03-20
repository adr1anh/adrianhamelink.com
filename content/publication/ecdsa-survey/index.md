---
title: "A Survey of Threshold ECDSA"

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here 
# and it will be replaced with their full name and linked to their profile.
authors:
- Jean-Philippe Aumasson (Taurus SA)
- admin
- Omer Schlomovits (ZenGo)

# Author notes (optional)
# author_notes:
# - "Equal contribution"
# - "Equal contribution"

date: "2020-10-01T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["3"]

# Publication name and optional abbreviated publication name.
# publication: In ePrint
# publication_short: ePrint

abstract: |
    Threshold signing research progressed a lot in the last three years, especially for ECDSA, which is less MPC-friendly than Schnorr-based signatures such as EdDSA. This progress was mainly driven by blockchain applications, and boosted by breakthrough results concurrently published by Lindell and by Gennaro & Goldfeder. Since then, several research teams published threshold signature schemes with different features, design trade-offs, building blocks, and proof techniques. Furthermore, threshold signing is now deployed within major organizations to protect large amounts of digital assets. Researchers and practitioners therefore need a clear view of the research state, of the relative merits of the protocols available, and of the open problems, in particular those that would address "real-world" challenges.

    This survey therefore proposes to:
    1) describe threshold signing and its building blocks in a general, unified way, based on the extended arithmetic black-box formalism (ABB+)
    2) review the state-of-the-art threshold signing protocols, highlighting their unique properties and comparing them in terms of security assurance and performance, based on criteria relevant in practice
    3) review the main open-source implementations available.

# Summary. An optional shortened abstract.
summary: A survey of various threshold ECDSA protocols under dishonest majority.

tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'https://eprint.iacr.org/2020/1390'
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# image:
#   caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/pLCdAaMFLTE)'
#   focal_point: ""
#   preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
# - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example
---

<!-- {{% callout note %}}
Click the *Cite* button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the *Slides* button to check out the example.
{{% /callout %}}

Supplementary notes can be added here, including [code, math, and images](https://wowchemy.com/docs/writing-markdown-latex/). -->
