#!/usr/bin/env bash
#
# This program will download all games on online-go.com for a specified user

BASE='https://online-go.com'
API='api/v1'

# Delay between SGF downloads (seconds). Increase if still getting rate limited.
RATE_DELAY=1
# Max retries per request
MAX_RETRIES=5
# Initial backoff in seconds (doubles on each retry)
BACKOFF=2

# Curl with retry + exponential backoff on failure or rate limiting (HTTP 429)
curl_with_retry() {
    local URL="$1"
    local OUTFILE="${2:-}"
    local ATTEMPT=1
    local WAIT=$BACKOFF

    while (( ATTEMPT <= MAX_RETRIES )); do
        if [[ -n "$OUTFILE" ]]; then
            HTTP_CODE=$(curl -s -w "%{http_code}" -o "$OUTFILE" "$URL")
        else
            RESPONSE=$(curl -s -w "\n%{http_code}" "$URL")
            HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
            BODY=$(echo "$RESPONSE" | head -n -1)
        fi

        if [[ "$HTTP_CODE" == "200" ]]; then
            [[ -z "$OUTFILE" ]] && echo "$BODY"
            return 0
        elif [[ "$HTTP_CODE" == "429" || "$HTTP_CODE" == "503" ]]; then
            echo -e "\nRate limited (HTTP $HTTP_CODE), waiting ${WAIT}s before retry $ATTEMPT/$MAX_RETRIES..." >&2
        else
            echo -e "\nHTTP $HTTP_CODE for $URL, retry $ATTEMPT/$MAX_RETRIES..." >&2
        fi

        sleep "$WAIT"
        WAIT=$(( WAIT * 2 ))
        ATTEMPT=$(( ATTEMPT + 1 ))
    done

    echo -e "\nFailed after $MAX_RETRIES retries: $URL" >&2
    [[ -n "$OUTFILE" ]] && rm -f "$OUTFILE"
    return 1
}

# Get username and ID
echo -n 'What is your OGS Username? '
read -r USERNAME

PLAYERINFO=$(curl_with_retry "$BASE/$API/players?username=$USERNAME") || {
    echo "Failed to reach OGS API." >&2
    exit 1
}

PLAYERID=$(echo "$PLAYERINFO" | jq '.results[0].id')
RETURNED_USERNAME=$(echo "$PLAYERINFO" | jq -r '.results[0].username')

if [[ "$RETURNED_USERNAME" != "$USERNAME" ]]; then
    echo "Username not found or mismatch (got: $RETURNED_USERNAME)" >&2
    exit 1
fi

echo "Found player: $USERNAME (ID: $PLAYERID)"

# Retrieve all games via pagination, extracting id + player names in one pass.
# The games list response already includes players, so no per-game API call needed.
# Each entry in GAMES will be: "<id> <white> <black>"
PAGEURL="$BASE/$API/players/$PLAYERID/games"
GAMES=()

while [[ "$PAGEURL" != "null" && -n "$PAGEURL" ]]; do
    echo "Fetching page: $PAGEURL"
    GAMELIST=$(curl_with_retry "$PAGEURL") || {
        echo "Failed to fetch game list page: $PAGEURL, stopping pagination." >&2
        break
    }

    if [[ -z "${NUMGAMES+x}" ]]; then
        NUMGAMES=$(echo "$GAMELIST" | jq '.count')
        echo "Total games: $NUMGAMES"
    fi

    while IFS= read -r line; do
        GAMES+=("$line")
    done < <(echo "$GAMELIST" | jq -r '.results[] | "\(.id) \(.players.white.username // "unknown") \(.players.black.username // "unknown")"')

    PAGEURL=$(echo "$GAMELIST" | jq -r '.next // "null"')

    # Small delay between pagination requests too
    sleep 0.5
done

echo "Collected ${#GAMES[@]} game entries."

# Progress bar
progress_bar() {
    local COUNT=$1 MAX=$2 MAXLENGTH=$3
    local BARCOUNT=$(( COUNT * MAXLENGTH / MAX ))
    printf "\r["
    printf '%0.s#' $(seq 1 "$BARCOUNT") 2>/dev/null || true
    printf '%0.s ' $(seq 1 $(( MAXLENGTH - BARCOUNT ))) 2>/dev/null || true
    printf "](%d/%d)" "$COUNT" "$MAX"
}

# Download each SGF
COUNT=1
TOTAL=${#GAMES[@]}
FAILED=0

for ENTRY in "${GAMES[@]}"; do
    read -r ID WHITE BLACK <<< "$ENTRY"

    progress_bar "$COUNT" "$TOTAL" 40
    COUNT=$(( COUNT + 1 ))

    # Sanitize names to avoid bad filenames
    WHITE=${WHITE//[^a-zA-Z0-9_.-]/_}
    BLACK=${BLACK//[^a-zA-Z0-9_.-]/_}

    FILENAME="${ID}-${WHITE}-${BLACK}.sgf"

    if [[ ! -e "$FILENAME" ]]; then
        if ! curl_with_retry "$BASE/$API/games/$ID/sgf" "$FILENAME"; then
            echo -e "\nFailed to download SGF for game $ID after retries, skipping." >&2
            FAILED=$(( FAILED + 1 ))
        fi
        # Delay between downloads to avoid rate limiting
        sleep "$RATE_DELAY"
    fi
done

echo -e "\nDone. $TOTAL games processed, $FAILED failed."
