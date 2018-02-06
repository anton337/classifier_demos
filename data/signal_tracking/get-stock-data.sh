#!/bin/bash

SYMBOL=$1
if [[ -z $SYMBOL ]]; then
  echo "Please enter a SYMBOL as the first parameter to this script"
  exit
fi
echo "Downloading quotes for $SYMBOL"


function log () {
  # To remove logging comment echo statement and uncoment the :
  echo $1
  # :
}

START_DATE=0
END_DATE=$(date +%s)

# Store the cookie in a temp file
cookieJar=$(mktemp)

# Get the crumb value
function getCrumb () {
  echo -en "$(curl -s --cookie-jar $cookieJar $1)" | tr "}" "\n" | grep CrumbStore | cut -d':' -f 3 | sed 's+"++g'
}

# TODO If crumb is blank then we probably don't have a valid symbol
URL="https://finance.yahoo.com/quote/$SYMBOL/?p=$SYMBOL"
log $URL
crumb=$(getCrumb $URL)
log $crumb
log "CRUMB: $crumb"
if [[ -z $crumb ]]; then
  echo "Error finding a valid crumb value"
  exit
fi


# Build url with SYMBOL, START_DATE, END_DATE
BASE_URL="https://query1.finance.yahoo.com/v7/finance/download/$SYMBOL?period1=$START_DATE&period2=$END_DATE&interval=1d&events=history"
log $BASE_URL

# Add the crumb value
URL="$BASE_URL&crumb=$crumb"
log "URL: $URL"

# Download to 
curl -s --cookie $cookieJar  $URL > $SYMBOL.csv

echo "Data downloaded to $SYMBOL.csv"
