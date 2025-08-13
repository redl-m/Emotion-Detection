class AttentionChart {
    /**
     * @param {string} selector CSS selector for the chart container.
     */
    constructor(selector) {
        this.selector = selector;
        this.chart = {}; // To hold D3 components
    }

    /**
     * Initializes the chart, setting up the SVG, scales, and axes.
     */
    init() {
        const c = this.chart; // shorthand

        const selectorStr = typeof this.selector === 'object'
            ? this.selector.chartSelector
            : this.selector;

        const container = d3.select(selectorStr);

        if (container.empty()) {
            console.error(`AttentionChart Error: Container not found for selector: ${selectorStr}`);
            return;
        }

        const margin = {top: 20, right: 20, bottom: 30, left: 50};
        const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
        const height = 200 - margin.top - margin.bottom;

        container.select('svg').remove(); // Clear previous

        c.svg = container.append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Scales
        c.x = d3.scaleTime().range([0, width]);
        c.y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        // Axes
        c.xAxis = c.svg.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${height})`);
        c.yAxis = c.svg.append('g').attr('class', 'y-axis').call(d3.axisLeft(c.y).ticks(5).tickFormat(d3.format('.0%')));

        // Line generator and path element
        c.line = d3.line().x(d => c.x(d.timestamp)).y(d => c.y(d.engagement)).curve(d3.curveMonotoneX);
        c.path = c.svg.append('path').attr('class', 'attention-line').style('stroke', '#1f77b4').style('fill', 'none').style('stroke-width', 2);
    }

    /**
     * Updates/renders the chart with a new set of data.
     * @param {object[]} history - Array of data points [{timestamp, engagement}].
     * @param {number} transitionDuration - Duration for transitions in ms.
     */
    update(history, transitionDuration = 100) {
        if (!this.chart.svg) return;

        // Filter out invalid points
        const validHistory = (history || []).filter(d =>
            d.timestamp instanceof Date &&
            !isNaN(d.timestamp) &&
            typeof d.engagement === 'number' &&
            !isNaN(d.engagement)
        );

        if (validHistory.length < 2) {
            this.clear();
            return;
        }

        const c = this.chart;
        c.x.domain(d3.extent(validHistory, d => d.timestamp));

        // Redraw axes and line
        c.xAxis.transition().duration(transitionDuration)
            .call(d3.axisBottom(c.x).ticks(5));

        c.path.datum(validHistory)
            .transition().duration(transitionDuration)
            .attr('d', c.line);
    }


    /**
     * Clears all data from the chart.
     */
    clear() {
        if (this.chart.path) {
            this.chart.path.datum([]).attr('d', this.chart.line);
        }
    }
}