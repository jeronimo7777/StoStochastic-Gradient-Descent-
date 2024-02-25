In this task we perform the Stochastic Gradient Descent to estimate the coefficients of a linear regression model. We used a data set from https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime that contains observations on crime per 100,000 inhabitants in US regions. The model that we try to estimate is :

𝑉𝑖𝑜𝑙𝑒𝑛𝑡𝐶𝑟𝑖𝑚𝑒𝑠𝑃𝑒𝑟𝑃𝑜𝑝= 𝛽1𝑚𝑒𝑑𝐼𝑛𝑐𝑜𝑚𝑒+ 𝛽2𝑤ℎ𝑖𝑡𝑒𝑃𝑒𝑟𝐶𝑎𝑝+ 𝛽3𝑏𝑙𝑎𝑐𝑘𝑃𝑒𝑟𝐶𝑎𝑝+𝛽4𝐻𝑖𝑠𝑝𝑃𝑒𝑟𝐶𝑎𝑝+𝛽5𝑁𝑢𝑚𝑈𝑛𝑑𝑒𝑟𝑃𝑜𝑣+ 𝛽6𝑃𝑐𝑡𝑈𝑛𝑒𝑚𝑝𝑙𝑜𝑦𝑒𝑑+ 𝛽7𝐻𝑜𝑢𝑠𝑉𝑎𝑐𝑎𝑛𝑡+ 𝛽8𝑀𝑒𝑑𝑅𝑒𝑛𝑡+ 𝛽9𝑁𝑢𝑚𝑆𝑡𝑟𝑒𝑒𝑡+ 𝛽0
