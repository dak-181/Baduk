#!/usr/bin/bash
# 
# This program will download all games on online-go.com for a specified user

BASE='https://online-go.com/'
API='api/v1'

# Get username and ID
echo 'What is your OGS Username? '
read USERNAME
PLAYERINFO=$(curl $BASE$API'/players?username='$USERNAME 2> /dev/null | jq '.results[0]')
PLAYERID=$(echo $PLAYERINFO | jq '.id')
echo $PLAYERID

# Verify the ID is correct
if [[ $(echo $PLAYERINFO | jq '.username' | sed 's/"//g') != "$USERNAME" ]]; then
    echo "Invalid ID or username!"
    exit 1
fi

# Retrieve Game IDs
PAGEURL=$BASE$API'/players/'$PLAYERID'/games'
GAMELIST=$(curl $PAGEURL 2> /dev/null)
NUMGAMES=$(echo $GAMELIST | jq '.count')
echo "Downloading $NUMGAMES games..."
IDS=$(echo $GAMELIST | jq '.results[] | .id')' '

while [[ $(echo $GAMELIST | jq '.next') != null ]]; do
    PAGEURL=$(echo $GAMELIST | jq '.next' | sed 's/"//g')
    echo $PAGEURL
    GAMELIST=$(curl $PAGEURL 2> /dev/null)
    IDS+=$(echo $GAMELIST | jq '.results[] | .id')' '
done

# For some eye candy to display how far along we are in downloading
progress_bar()
{
    local COUNT=$1
    local MAX=$2
    local MAXLENGTH=$3
    echo -ne "\r"
    echo -ne "["
    local BARCOUNT=$(($COUNT  * $MAXLENGTH / $MAX))
    perl -E "print '#' x $BARCOUNT"
    perl -E "print ' ' x ($MAXLENGTH - $BARCOUNT)"
    echo -ne "]($1/$2)"
}

# Download each game and give it a filename
COUNT=1
for ID in $IDS; do
    progress_bar $COUNT $NUMGAMES 40
    COUNT=$(($COUNT + 1))
    GAMEINFO=$(curl $BASE$API'/games/'$ID'/' 2> /dev/null)
    WHITE=$(echo $GAMEINFO | jq '.players.white.username' | sed 's/"//g')
    BLACK=$(echo $GAMEINFO | jq '.players.black.username' | sed 's/"//g')
    FILENAME="${ID}-${WHITE}-${BLACK}.sgf"
    if [[ ! -e "$FILENAME" ]]; then
        curl -o "$FILENAME" $BASE$API'/games/'$ID'/sgf' 2> /dev/null
    fi
done
