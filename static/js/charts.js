class ChartManager {


    /**
     * @param {object} config Configuration for the chart manager.
     * @param {string} config.barChartSelector CSS selector for the bar chart container.
     * @param {string} config.timeSeriesSelector CSS selector for the time-series chart container.
     * @param {string} config.legendSelector CSS selector for the legend container.
     * @param {string[]} config.emotions Array of emotion labels.
     * @param {function} config.colors d3.scaleOrdinal color scale.
     */
    constructor(config) {
        this.config = config;
        this.charts = {}; // To hold SVG elements, scales, etc.
    }


    /**
     * Sets up the chart areas, scales, and axes.
     */
    init() {
        this._setupLegend();
        this._setupBarChart();
        this._setupTimeSeriesChart();
    }


    /**
     * Sets up the chart legends.
     * @private
     */
    _setupLegend() {
        const legend = d3.select(this.config.legendSelector);
        legend.html(''); // Clear existing
        this.config.emotions.forEach((name, i) => {
            const item = legend.append('div').attr('class', 'legend-item');
            item.append('div').attr('class', 'legend-color').style('background-color', this.config.colors(i));
            item.append('span').text(name);
        });
    }


    /**
     * Builds the bar chart's SVG, axes and bars with default values.
     * @private
     */
    _setupBarChart() {
        const c = this.charts; // shorthand
        const margin = { top: 5, right: 20, bottom: 30, left: 65 };
        const container = d3.select(this.config.barChartSelector);
        const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
        const height = 180 - margin.top - margin.bottom;

        c.bcSvg = container.append('svg').attr('width', '100%').attr('height', '100%')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

        c.yB = d3.scaleBand().domain(this.config.emotions).range([0, height]).padding(0.2);
        c.xB = d3.scaleLinear().domain([0, 1]).range([0, width]);

        c.bcSvg.append('g').attr('class', 'y-axis-b').call(d3.axisLeft(c.yB).tickSize(0)).select('.domain').remove();
        c.bcSvg.append('g').attr('class', 'x-axis-b').attr('transform', `translate(0, ${height})`)
            .call(d3.axisBottom(c.xB).ticks(5).tickFormat(d3.format('.0%')));

        c.bars = c.bcSvg.selectAll('.bar-b').data(this.config.emotions.map(e => ({ emotion: e, value: 0 }))).enter()
            .append('rect').attr('class', 'bar-b').attr('y', d => c.yB(d.emotion)).attr('x', 0)
            .attr('height', c.yB.bandwidth()).attr('width', 0).attr('rx', 3).style('fill', (d, i) => this.config.colors(i));
    }


    /**
     * Sets up the time series chart's SVG, axes and bars with default values.
     * @private
     */
    _setupTimeSeriesChart() {
        const c = this.charts; // shorthand
        const margin = { top: 5, right: 20, bottom: 30, left: 40 };
        const container = d3.select(this.config.timeSeriesSelector);
        const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
        const height = 180 - margin.top - margin.bottom;

        c.tsSvg = container.append('svg').attr('width', '100%').attr('height', '100%')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        c.xTS = d3.scaleTime().range([0, width]);
        c.yTS = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        c.tsSvg.append('g').attr('class', 'x-axis-ts').attr('transform', `translate(0,${height})`);
        c.tsSvg.append('g').attr('class', 'y-axis-ts').call(d3.axisLeft(c.yTS).ticks(5).tickFormat(d3.format('.0%')));

        c.lineGen = this.config.emotions.map((_, i) => d3.line().x(d => c.xTS(d.timestamp)).y(d => c.yTS(d.probs[i])).curve(d3.curveMonotoneX));
        c.lines = c.tsSvg.selectAll('.line-ts').data(this.config.emotions).enter().append('path')
            .attr('class', 'line-ts').style('stroke', (d, i) => this.config.colors(i)).style('fill', 'none').style('stroke-width', 2.5);
    }


    /**
     * Updates the charts with new data.
     * @param {object[]} history - Array of data points {timestamp, probs}.
     * @param {number} transitionDuration - Duration for transitions in ms.
     */
    update(history, transitionDuration = 100) {
        const c = this.charts;

        // Update Bar Chart
        const latestProbs = history.length > 0 ? history[history.length - 1].probs : new Array(this.config.emotions.length).fill(0);
        const barData = this.config.emotions.map((e, i) => ({ emotion: e, value: latestProbs[i] }));
        c.bars.data(barData).transition().duration(transitionDuration).attr('width', d => c.xB(d.value));

        // Update Time Series Chart
        if (history.length < 2) {
            c.lines.attr('d', null);
            return;
        }
        c.xTS.domain(d3.extent(history, d => d.timestamp));
        c.tsSvg.select('.x-axis-ts').transition().duration(transitionDuration).call(d3.axisBottom(c.xTS).ticks(5));
        c.lines.data(this.config.emotions).attr('d', (d, i) => c.lineGen[i](history));
    }


    /**
     * Clears all data from the charts.
     */
    clear() {
        const emptyData = this.config.emotions.map(e => ({ emotion: e, value: 0 }));
        this.charts.bars.data(emptyData).transition().duration(100).attr('width', 0);
        this.charts.lines.attr('d', null);
    }


    /**
     * A static utility method to create a standalone donut chart.
     * @param {string} selector - CSS selector for the container.
     * @param {number[]} distribution - Array of probability values.
     * @param {object} config - The same config object used by the manager instance.
     */
    static createDonut(selector, distribution, config) {
        const data = distribution.map((value, i) => ({ value, name: config.emotions[i] })).filter(d => d.value > 0);
        const width = 80, height = 80, margin = 5;
        const radius = Math.min(width, height) / 2 - margin;

        d3.select(selector).select('svg').remove(); // Clear previous
        const svg = d3.select(selector).append("svg")
            .attr("class", "donut-chart-svg")
            .attr("viewBox", `0 0 ${width} ${height}`)
            .append("g")
            .attr("transform", `translate(${width / 2},${height / 2})`);

        const pie = d3.pie().value(d => d.value).sort(null);
        const arc = d3.arc().innerRadius(radius * 0.5).outerRadius(radius);

        svg.selectAll('path').data(pie(data)).enter().append('path')
            .attr('d', arc)
            .attr('fill', d => config.colors(config.emotions.indexOf(d.data.name)));
    }
}